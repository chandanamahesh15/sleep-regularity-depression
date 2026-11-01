"""Command line entry point: ``sleeprisk run`` / ``sleeprisk synth``."""

from __future__ import annotations

import argparse
import logging
import sys

from sleeprisk.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sleeprisk", description="Sleep regularity -> depression risk pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the full experiment")
    run_p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    run_p.add_argument("--data-dir", default=None, help="Override data.base_dir")
    run_p.add_argument("--out", default=None, help="Override data.output_dir")
    run_p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Dotted config override")
    run_p.add_argument("-v", "--verbose", action="store_true")

    syn = sub.add_parser("synth", help="Write a synthetic StudentLife-format dataset")
    syn.add_argument("--out", default="data/synthetic/dataset")
    syn.add_argument("--n-users", type=int, default=46)
    syn.add_argument("--days", type=int, default=66)
    syn.add_argument("--seed", type=int, default=7)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd == "synth":
        from sleeprisk.synthetic import write_synthetic_studentlife

        path = write_synthetic_studentlife(args.out, args.n_users, args.days, args.seed)
        print(f"Synthetic dataset written to {path}")
        return 0

    overrides = list(args.set)
    if args.data_dir:
        overrides.append(f"data.base_dir={args.data_dir}")
    if args.out:
        overrides.append(f"data.output_dir={args.out}")
    cfg = load_config(args.config, overrides)

    from sleeprisk.pipeline import run

    metrics = run(cfg)
    print(f"Report: {cfg['data']['output_dir']}/report.md")
    best = max(metrics["subject_cv"].items(), key=lambda kv: kv[1]["summary"]["auroc"]["mean"])
    print(f"Best subject-level AUROC: {best[0]} = {best[1]['summary']['auroc']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
