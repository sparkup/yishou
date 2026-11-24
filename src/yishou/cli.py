"""Count to ten on one hand: record samples, train and test the classifier, run the demo."""

import argparse
from pathlib import Path

from yishou import __version__

DEFAULT_DATA = Path("data/samples.csv")
DEFAULT_MODEL = Path("models/latest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yishou", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="record labelled samples from the webcam")
    collect.add_argument("--data", type=Path, default=DEFAULT_DATA)
    collect.add_argument("--camera", type=int, default=0)

    train = commands.add_parser("train", help="train the classifier on recorded samples")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--epochs", type=int, default=80)

    evaluate = commands.add_parser(
        "evaluate", help="score a model on recordings it was not trained on"
    )
    evaluate.add_argument("data", type=Path, help="samples recorded with `yishou collect --data`")
    evaluate.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="local directory or Hugging Face repo id"
    )

    demo = commands.add_parser("demo", help="run the live webcam demo")
    demo.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="local directory or Hugging Face repo id"
    )
    demo.add_argument("--camera", type=int, default=0)
    demo.add_argument("--threshold", type=float, default=0.6)
    demo.add_argument("--record", type=Path, default=None, help="save the preview as an mp4")

    push = commands.add_parser("push", help="publish a trained model to Hugging Face")
    push.add_argument("repo_id", help="for example your-username/yishou")
    push.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    push.add_argument("--private", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Imports are deferred so that `train` and `push` never touch the camera stack,
    # and `--help` stays instant.
    match args.command:
        case "collect":
            from yishou.collect import collect

            collect(args.data, args.camera)
        case "train":
            from yishou.train import train

            train(args.data, args.out, epochs=args.epochs)
        case "evaluate":
            from yishou.evaluate import evaluate

            evaluate(args.model, args.data)
        case "demo":
            from yishou.demo import demo

            demo(args.model, args.camera, args.threshold, args.record)
        case "push":
            from yishou.hub import push

            print(push(args.model, args.repo_id, args.private))


if __name__ == "__main__":
    main()
