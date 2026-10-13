"""Command line entry point: collect, train."""

import argparse
from pathlib import Path

DEFAULT_DATA = Path("data/samples.csv")
DEFAULT_MODEL = Path("models/latest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yishou", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="record labelled samples from the webcam")
    collect.add_argument("--data", type=Path, default=DEFAULT_DATA)
    collect.add_argument("--camera", type=int, default=0)

    train = commands.add_parser("train", help="train the classifier on recorded samples")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--epochs", type=int, default=80)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Imports are deferred so that `train` never touches the camera stack, and
    # `--help` stays instant.
    match args.command:
        case "collect":
            from yishou.collect import collect

            collect(args.data, args.camera)
        case "train":
            from yishou.train import train

            train(args.data, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
