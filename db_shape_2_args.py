import miopUtil.shapeConvert as shapeConvert
import argparse
import sys

def Solve():
    # Parse command name to determine data type
    parser = argparse.ArgumentParser(description='convert db shape to args',
                                     add_help=False)
    parser.add_argument('--solver', type=str, default='',
                        help='the shape of the solver')
    parser.add_argument('--kernel', type=str, default='',
                        help='the shape of the kernel')
    
    args = parser.parse_args()

    pd = shapeConvert.ProblemDescription()
    if args.solver != '':
        problem = pd.ufdbDeserialize(args.solver)
        miargs = pd.Problem2MIArgs(problem)
        cmds, conv_type = shapeConvert.GetArgument(miargs)
    elif args.kernel != '':
        problem = pd.udbDeserialize(args.kernel)
        miargs = pd.Problem2MIArgs(problem)
        cmds, conv_type = shapeConvert.GetArgument(miargs)

    print(f"{conv_type}", end=' ')
    for cmd in cmds:
        print(cmd, end=' ')
    print()

if __name__ == "__main__":
    try:
        Solve()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)