"""``brisca`` command line: run tournaments and publish leaderboards."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from brisca.report import leaderboard, update_readme
from brisca.store import ResultsStore
from brisca.tournament import TournamentConfig, run_tournament

log = logging.getLogger("brisca")
DEFAULT_DB = Path("results/brisca.duckdb")


def _tournament(args: argparse.Namespace) -> None:
    config = TournamentConfig.from_toml(args.config)
    pairs = len(config.agents) * (len(config.agents) - 1) // 2
    log.info(
        "%d agents, %d pairings, %d games", len(config.agents), pairs, pairs * config.deals * 2
    )
    start = time.perf_counter()
    records = run_tournament(config, workers=args.workers)
    store = ResultsStore(args.db)
    tournament_id = store.save(config, records)
    store.close()
    log.info(
        "tournament %d saved to %s in %.0fs", tournament_id, args.db, time.perf_counter() - start
    )


def _report(args: argparse.Namespace) -> None:
    store = ResultsStore(args.db)
    tournament = args.tournament or store.latest_tournament()
    content = leaderboard(store, tournament, anchor=args.anchor, bootstrap=args.bootstrap)
    if args.export_games:
        store.export_games(tournament, args.export_games)
    store.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content)
    if args.readme:
        update_readme(args.readme, content)
    if not args.out and not args.readme:
        print(content)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(prog="brisca", description=__doc__)
    sub = parser.add_subparsers(required=True)

    run = sub.add_parser("tournament", help="run a round-robin tournament from a TOML config")
    run.add_argument("config", type=Path)
    run.add_argument("--db", type=Path, default=DEFAULT_DB)
    run.add_argument("--workers", type=int, default=None)
    run.set_defaults(func=_tournament)

    rep = sub.add_parser("report", help="print or publish the leaderboard for a tournament")
    rep.add_argument("--db", type=Path, default=DEFAULT_DB)
    rep.add_argument("--tournament", type=int, default=None, help="defaults to the latest")
    rep.add_argument("--anchor", default="random", help="agent rated 0 (default: random)")
    rep.add_argument("--bootstrap", type=int, default=500)
    rep.add_argument("--out", type=Path, default=None)
    rep.add_argument("--readme", type=Path, default=None)
    rep.add_argument("--export-games", type=Path, default=None, help="write games to Parquet")
    rep.set_defaults(func=_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
