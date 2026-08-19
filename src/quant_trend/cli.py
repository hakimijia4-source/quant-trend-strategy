from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .cognition import assert_point_in_time_safe, load_event_files, load_theses
from .config import StrategyConfig, load_config
from .demo import run_demo
from .environment import build_offline_trajectories
from .features import build_feature_frame
from .labels import build_labeled_dataset
from .sources import collect_market, collect_public_events


def _default_path(config: StrategyConfig, name: str) -> Path | None:
    candidate = config.data_dir / name
    return candidate if candidate.exists() else None


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def command_collect_market(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    symbols = args.symbols.split(",") if args.symbols else None
    paths = collect_market(
        config,
        symbols=symbols,
        start=args.start,
        end=args.end,
        provider=args.provider,
    )
    _print({"market_files": [str(path) for path in paths]})


def command_collect_events(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    start = args.start or str(config.get("market", "start"))
    end = args.end or str(config.get("market", "end"))
    paths = collect_public_events(config, start, end)
    _print({"event_files": [str(path) for path in paths]})


def command_build(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    thesis = Path(args.theses) if args.theses else _default_path(config, "theses.jsonl")
    manual = Path(args.manual_events) if args.manual_events else _default_path(config, "manual_events.csv")
    features = build_feature_frame(config, thesis_path=thesis, manual_events=manual)
    labeled = build_labeled_dataset(config, features)
    trajectories = build_offline_trajectories(labeled, config)
    _print(
        {
            "feature_rows": len(features),
            "labeled_rows": int(labeled["trend_label"].notna().sum()),
            "trajectories": int(trajectories["trajectory_id"].nunique()),
            "trajectory_rows": len(trajectories),
        }
    )


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    try:
        from .trainer import train_walk_forward
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                'PyTorch is required. Install the training dependencies with: pip install -e ".[train]"'
            ) from exc
        raise
    results = train_walk_forward(config, max_folds=args.max_folds)
    _print(
        {
            "folds": len(results),
            "latest_metrics": results[-1]["test_metrics"] if results else {},
            "reports_dir": str(config.reports_dir),
        }
    )


def command_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    thesis = Path(args.theses) if args.theses else _default_path(config, "theses.jsonl")
    manual = Path(args.manual_events) if args.manual_events else _default_path(config, "manual_events.csv")
    theses = load_theses(thesis)
    events = load_event_files(config.data_dir / "raw" / "events", manual)
    assert_point_in_time_safe(events, theses)
    retrospective = sum(item.is_retrospective for item in theses)
    _print(
        {
            "status": "ok",
            "target_symbol": config.target_symbol,
            "events": len(events),
            "theses": len(theses),
            "retrospective_theses_excluded": retrospective,
            "live_trading_enabled": False,
        }
    )


def command_demo(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _print(run_demo(config, sessions=args.sessions, train=not args.skip_train))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-trend",
        description="Point-in-time data collection and offline hierarchical RL training.",
    )
    parser.add_argument("--config", default="config/strategy.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_market_parser = subparsers.add_parser("collect-market")
    collect_market_parser.add_argument("--symbols", help="Comma-separated symbols; default is target plus context")
    collect_market_parser.add_argument("--start")
    collect_market_parser.add_argument("--end")
    collect_market_parser.add_argument("--provider", choices=["alpaca", "massive"])
    collect_market_parser.set_defaults(func=command_collect_market)

    collect_event_parser = subparsers.add_parser("collect-events")
    collect_event_parser.add_argument("--start")
    collect_event_parser.add_argument("--end")
    collect_event_parser.set_defaults(func=command_collect_events)

    build = subparsers.add_parser("build-dataset")
    build.add_argument("--theses")
    build.add_argument("--manual-events")
    build.set_defaults(func=command_build)

    train = subparsers.add_parser("train")
    train.add_argument("--max-folds", type=int)
    train.set_defaults(func=command_train)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--theses")
    validate.add_argument("--manual-events")
    validate.set_defaults(func=command_validate)

    demo = subparsers.add_parser("demo")
    demo.add_argument("--sessions", type=int, default=180)
    demo.add_argument("--skip-train", action="store_true")
    demo.set_defaults(func=command_demo)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

