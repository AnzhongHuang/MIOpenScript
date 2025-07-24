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
    for line in lines:
        #print(line.strip())
        if line.strip() and not line.startswith('#'):
            key, value = line.split('=')
            params[key.strip()] = value.strip()
    # print(f"Parsed parameters")
    return params

def PrintCmds(cmds, conv_type, output=sys.stdout):
    #print(f"{conv_type}", end=' ')
    #for cmd in cmds:
    #    print(cmd, end=' ')
    #print()
    output.write(f"MIOpenDriver {conv_type} ")
    for cmd in cmds:
        output.write(cmd + ' ')
    output.write('\n')


def Solve():
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
        params = ParseUfdb(args.solver_file)
        output_file = args.output if args.output else 'output.txt'
        with open(output_file, 'w') as f:
            for key, value in params.items():
                # print(f"{key}={value}")
                problem = pd.ufdbDeserialize(key)
                miargs = pd.Problem2MIArgs(problem)
                cmds, conv_type = shapeConvert.GetArgument(miargs)
                print(f"# {key}", file=f)
                PrintCmds(cmds, conv_type, f)
    
if __name__ == "__main__":
    try:
        Solve()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)