"""The command line parser."""

import pytest

from yishou import __version__
from yishou.cli import build_parser


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    assert capsys.readouterr().out.strip() == f"yishou {__version__}"


def test_evaluate_needs_recordings() -> None:
    args = build_parser().parse_args(["evaluate", "theirs.csv", "--model", "someone/yishou"])
    assert (str(args.data), args.model) == ("theirs.csv", "someone/yishou")
