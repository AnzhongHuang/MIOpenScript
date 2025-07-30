import miopUtil.shapeConvert as shapeConvert
import argparse
import sys
from miopUtil.MIArgs import MiopenDataType

def ParseUfdb(file_path):
    """
    Parse the ufdb.txt file to extract parameters.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    params = {}
    count = 0
    for line in lines:
        #print(line.strip())
        if line.strip() and not line.startswith('#'):
            key, value = line.split('=')
            count += 1
            params[key.strip()] = value.strip()
    # print(f"Parsed parameters")
    return params, count

def PrintCmds(cmds, conv_type, output=sys.stdout):
    #print(f"{conv_type}", end=' ')
    #for cmd in cmds:
    #    print(cmd, end=' ')
    #print()
    output.write(f"MIOpenDriver {conv_type} ")
    for cmd in cmds:
        output.write(cmd + ' ')
    output.write('\n')

solver_map = {}

class Statistics:
    def __init__(self, execTime=0.0, gflops=0.0, count=0, best_count=0):
        self.execTime = execTime
        self.gflops = gflops
        self.support_count = count
        self.best_count = best_count
        self.average_gflops = gflops / count if count > 0 else 0.0


def StatisticsSolver(solvers, problem):

    global solver_map
    if problem.spatial_dims == 2:
        flopCnt = 2 * problem.in_batch_size * problem.out_channels * problem.out_height * problem.out_width * \
                (problem.in_channels // problem.group_count) * problem.weights_height * problem.weights_width
    if problem.spatial_dims == 3:
        flopCnt = 2 * problem.in_batch_size * problem.out_channels * problem.out_depth * problem.out_height * \
            problem.out_width * (problem.in_channels // problem.group_count) * problem.weights_depth * \
            problem.weights_height * problem.weights_width
    # solver_map['solver_name'] = (execTime, workspaceSize, algorithm, gflops)

    best_solver_time = 10000.0
    best_solver_name = 'unknown'
    for solver in solvers:
        # string before : is the solver name
        solver_name, result= solver.split(':')
        params = result.split(',')
        # params[0] is a float time in ms
        execTime = float(params[0])
        if (execTime == 0.0):
            print(f"Warning: {solver_name} execution time is 0.0 ms.", file=sys.stderr)
        # params[1] is a int
        workspaceSize = int(params[1])
        # params[2] is a string
        algorithm = params[2]

        if execTime < best_solver_time:
            best_solver_time = execTime
            best_solver_name = solver_name

          # Avoid division by zero in GFLOPS calculation, it should be a bug in db file.
        if execTime == 0.0:
            execTime = 0.001

        gflops = flopCnt / (execTime * 1e6)
        if solver_name in solver_map:
            solver_map[solver_name].execTime += execTime
            solver_map[solver_name].gflops += gflops
            solver_map[solver_name].support_count += 1
        else:
            solver_map[solver_name] = Statistics(
                execTime=execTime, gflops=gflops, count=1, best_count=0)

    # bump the best solver count
    if best_solver_name in solver_map:
        solver_map[best_solver_name].best_count += 1

class PerfDbArgs:
    def __init__(self, mode=1, datatype=1, dimension=2, direction=1, in_batch=-1, in_channel=-1,
                 in_width=-1, in_height=-1, in_depth=-1, filter_width=-1, filter_height=-1,
                 filter_depth=-1, out_channel=-1, pad_width=0, pad_height=0, pad_depth=0,
                 step_width=1, step_height=1, step_depth=-1, dilation_width=1,
                 dilation_height=1, dilation_depth=-1, groups=1, in_layout=1,
                 filter_layout=1, out_layout=1, bias=0):
        self.mode = mode
        self.datatype = datatype
        self.dimension = dimension
        self.direction = direction
        self.in_batch = in_batch
        self.in_channel = in_channel
        self.in_width = in_width
        self.in_height = in_height
        self.in_depth = in_depth
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_depth = filter_depth
        self.out_channel = out_channel
        self.pad_width = pad_width
        self.pad_height = pad_height
        self.pad_depth = pad_depth
        self.step_width = step_width
        self.step_height = step_height
        self.step_depth = step_depth
        self.dilation_width = dilation_width
        self.dilation_height = dilation_height
        self.dilation_depth = dilation_depth
        self.groups = groups
        self.in_layout = in_layout
        self.filter_layout = filter_layout
        self.out_layout = out_layout
        self.bias = bias

    @staticmethod
    def PrintMIOpen(perfdb):
        def_val = PerfDbArgs()
        exec_str = "MIOpenDriver "
        if perfdb.mode == 1:
            exec_str += 'conv'
        else:
            # error
            raise ValueError("Unsupported mode in PerfDbArgs")
        if perfdb.datatype == 1:
            pass
        elif perfdb.datatype == 2:
            exec_str += 'fp16'
        elif perfdb.datatype == 3:
            exec_str += 'bfp16'
        else:
            raise ValueError("Unsupported datatype in PerfDbArgs")

        if perfdb.dimension != def_val.dimension:
            exec_str += f' --spatial_dim {perfdb.dimension}'

        exec_str += f' -F {perfdb.direction}'

        exec_str += f' -n {perfdb.in_batch}'
        exec_str += f' -c {perfdb.in_channel}'
        exec_str += f' -H {perfdb.in_height}'
        exec_str += f' -W {perfdb.in_width}'
        if perfdb.in_depth != def_val.in_depth:
            exec_str += f' --in_d {perfdb.in_depth}'

        exec_str += f' -k {perfdb.out_channel}'
        exec_str += f' -y {perfdb.filter_height}'
        exec_str += f' -x {perfdb.filter_width}'
        if perfdb.filter_depth != def_val.filter_depth:
            exec_str += f' --fil_d {perfdb.filter_depth}'
        if perfdb.pad_width != def_val.pad_width:
            exec_str += f' -p {perfdb.pad_width}'
        if perfdb.pad_height != def_val.pad_height:
            exec_str += f' -q {perfdb.pad_height}'
        if perfdb.pad_depth != def_val.pad_depth:
            exec_str += f' --pad_d {perfdb.pad_depth}'
        if perfdb.step_width != def_val.step_width:
            exec_str += f' -u {perfdb.step_width}'
        if perfdb.step_height != def_val.step_height:
            exec_str += f' -v {perfdb.step_height}'
        if perfdb.step_depth != def_val.step_depth:
            exec_str += f' --conv_stride_d {perfdb.step_depth}'
        if perfdb.dilation_width != def_val.dilation_width:
            exec_str += f' -l {perfdb.dilation_width}'
        if perfdb.dilation_height != def_val.dilation_height:
            exec_str += f' -j {perfdb.dilation_height}'
        if perfdb.dilation_depth != def_val.dilation_depth:
            exec_str += f' --dilation_d {perfdb.dilation_depth}'
        if perfdb.groups != def_val.groups:
            exec_str += f' -g {perfdb.groups}'
        if perfdb.mode == 1:
            exec_str += ' -m conv'
        if perfdb.in_layout != def_val.in_layout:
            if perfdb.dimension == 2:
                if perfdb.in_layout == 1:
                    pass
                elif perfdb.in_layout == 2:
                    exec_str += ' --in_layout NHWC'
                    exec_str += ' --out_layout NHWC'
                    exec_str += ' --fil_layout NHWC'
                else:
                    raise ValueError(f"Unsupported in_layout {perfdb.in_layout} for 2D convolution")
            elif perfdb.dimension == 3:
                if perfdb.in_layout == 1:
                    pass
                elif perfdb.in_layout == 2:
                    exec_str += ' --in_layout NDHWC'
                    exec_str += ' --out_layout NDHWC'
                    exec_str += ' --fil_layout NDHWC'
                else:
                    raise ValueError(f"Unsupported in_layout {perfdb.in_layout} for 3D convolution")
        exec_str += ' -t 1'
        print(exec_str)

def ParseROCmPerfDb(shape_str):
    params = shape_str.split('|')

    perfdb = PerfDbArgs(
        mode = int(params[0]),
        datatype = int(params[1]),
        dimension = int(params[2]),
        direction = int(params[3]),
        in_batch = int(params[4]),
        in_channel = int(params[5]),
        in_width = int(params[6]),
        in_height = int(params[7]),
        in_depth = int(params[8]),
        filter_width = int(params[9]),
        filter_height = int(params[10]),
        filter_depth = int(params[11]),
        out_channel = int(params[12]),
        pad_width = int(params[13]),
        pad_height = int(params[14]),
        pad_depth = int(params[15]),
        step_width = int(params[16]),
        step_height = int(params[17]),
        step_depth = int(params[18]),
        dilation_width = int(params[19]),
        dilation_height = int(params[20]),
        dilation_depth = int(params[21]),
        groups = int(params[22]),
        in_layout = int(params[23]),
        filter_layout = int(params[24]),
        out_layout = int(params[25]),
        bias = int(params[26]),
    )
    PerfDbArgs.PrintMIOpen(perfdb)

def Solve():
    global solver_map
    # Parse command name to determine data type
    parser = argparse.ArgumentParser(description='convert db shape to args',
                                     add_help=False)
    parser.add_argument('--solver', type=str, default='',
                        help='the shape of the solver')
    parser.add_argument('--kernel', type=str, default='',
                        help='the shape of the kernel')
    parser.add_argument('--solver_file', type=str, default='',
                        help='the ufdb.txt to parameters file')
    
    parser.add_argument('--output', type=str, default='',
                        help='parameters output file')
    
    parser.add_argument('--perf', type=str, default='',
                        help='the shape of Rocm performance db')
    args = parser.parse_args()

    pd = shapeConvert.ProblemDescription()
    if args.solver != '':
        problem = pd.ufdbDeserialize(args.solver)
        miargs = pd.Problem2MIArgs(problem)
        cmds, conv_type = shapeConvert.GetArgument(miargs)
        PrintCmds(cmds, conv_type)
    elif args.kernel != '':
        problem = pd.udbDeserialize(args.kernel)
        miargs = pd.Problem2MIArgs(problem)
        cmds, conv_type = shapeConvert.GetArgument(miargs)
        PrintCmds(cmds, conv_type)
    elif args.solver_file != '':
        params, count = ParseUfdb(args.solver_file)
        output_file = args.output if args.output else 'output.txt'
        with open(output_file, 'w') as f:
            for key, value in params.items():
                # print(f"{key}={value}")
                problem = pd.ufdbDeserialize(key)
                miargs = pd.Problem2MIArgs(problem)
                cmds, conv_type = shapeConvert.GetArgument(miargs)
                print(f"# {key}", file=f)
                PrintCmds(cmds, conv_type, f)

                solvers = value.split(';')
                StatisticsSolver(solvers=solvers, problem=problem)

        print(f"Database count: {len(params)}/{count}")
        # sort solver_map according to the best_count
        solver_map = dict(
            sorted(solver_map.items(), key=lambda item: item[1].best_count, reverse=True))

        # Print the statistics
        print(f"Statistics for {len(solver_map)} solvers:")
        print(f"{'Solver Name':<50} {'Total Exec Time (ms)':<25} {'GFLOPS':<20} "
              f"{'Support Count':<15} {'Select Count':<15} {'Avg GFLOPS':<15} {'Avg Exec Time (ms)':<15}")
        for solver_name, stats in solver_map.items():
            print(f"{solver_name:<50} {stats.execTime:<25.4f} {stats.gflops:<20.4f} "
                  f"{stats.support_count:<15} {stats.best_count:<15} "
                  f"{stats.average_gflops:<15.4f} {stats.execTime/stats.support_count:<15.4f}")
    elif args.perf != '':
        ParseROCmPerfDb(args.perf)

if __name__ == "__main__":
    try:
        Solve()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)