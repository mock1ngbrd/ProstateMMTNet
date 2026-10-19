"""Command-line entry point for evaluating ProstateMMTNet."""
import os

from utils.cli import build_test_parser, validate_inputs


def main(argv=None):
    parser = build_test_parser()
    args = parser.parse_args(argv)
    try:
        validate_inputs(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from evaluation import evaluate
    evaluate(args)


if __name__ == "__main__":
    main()
