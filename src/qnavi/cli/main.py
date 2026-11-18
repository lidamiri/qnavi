from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from qnavi.cli import commands
from qnavi.errors import QnaviError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qnavi",
        description="Q-learning and tabular reinforcement learning experiments",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    train = sub.add_parser("train", help="train an agent from a configuration file")
    train.add_argument("--config", required=True, help="path to a JSON configuration")
    train.add_argument("--output", default="runs", help="directory that receives the run")
    train.add_argument("--seed", type=int, default=None, help="override the configured seed")
    train.add_argument("--episodes", type=int, default=None, help="override the episode count")
    train.add_argument("--quiet", action="store_true", help="suppress progress lines")
    train.set_defaults(handler=commands.cmd_train)

    evaluate = sub.add_parser("evaluate", help="evaluate the greedy policy of a run")
    evaluate.add_argument("--run-dir", default="runs", help="run directory or parent of runs")
    evaluate.add_argument("--episodes", type=int, default=200)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--max-steps", type=int, default=0, help="0 uses the configured limit")
    evaluate.set_defaults(handler=commands.cmd_evaluate)

    render = sub.add_parser("render", help="print the greedy policy of a run")
    render.add_argument("--run-dir", default="runs", help="run directory or parent of runs")
    render.set_defaults(handler=commands.cmd_render)

    compare = sub.add_parser("compare", help="train every algorithm on one configuration")
    compare.add_argument("--config", required=True)
    compare.add_argument("--output", default="runs/compare")
    compare.add_argument("--seed", type=int, default=None)
    compare.add_argument("--episodes", type=int, default=None)
    compare.add_argument("--agents", default="", help="comma separated agent names")
    compare.add_argument("--quiet", action="store_true")
    compare.set_defaults(handler=commands.cmd_compare)

    sweep = sub.add_parser("sweep", help="sweep the learning rate of an agent")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--output", default="runs/sweep")
    sweep.add_argument("--seed", type=int, default=None)
    sweep.add_argument("--episodes", type=int, default=None)
    sweep.add_argument("--alphas", default="", help="comma separated learning rates")
    sweep.add_argument("--quiet", action="store_true")
    sweep.set_defaults(handler=commands.cmd_sweep)

    plot = sub.add_parser("plot", help="write learning curves for a run")
    plot.add_argument("--run-dir", default="runs")
    plot.add_argument("--window", type=int, default=50)
    plot.set_defaults(handler=commands.cmd_plot)

    listing = sub.add_parser("list", help="list registered components")
    listing.set_defaults(handler=commands.cmd_list)

    version = sub.add_parser("version", help="print the package version")
    version.set_defaults(handler=commands.cmd_version)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except QnaviError as exc:
        print(f"qnavi: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("qnavi: interrupted", file=sys.stderr)
        return 130
