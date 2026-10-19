"""Command-line entry point for training ProstateMMTNet."""
import os

from utils.cli import build_train_parser, validate_inputs


def main(argv=None):
    parser = build_train_parser()
    args = parser.parse_args(argv)
    try:
        validate_inputs(args, training=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from training import train
    train(args)


if __name__ == "__main__":
    main()
