import miopUtil.shapeConvert as shapeConvert
import argparse
import sys

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
from miopUtil.MIArgs import MiopenDataType
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

if __name__ == "__main__":
    try:
        Solve()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)