"""Command line entry point: collect."""

import argparse
from pathlib import Path

DEFAULT_DATA = Path("data/samples.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yishou", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="record labelled samples from the webcam")
    collect.add_argument("--data", type=Path, default=DEFAULT_DATA)
    collect.add_argument("--camera", type=int, default=0)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Imports are deferred so that `--help` stays instant.
    match args.command:
        case "collect":
            from yishou.collect import collect

            collect(args.data, args.camera)


if __name__ == "__main__":
    main()
