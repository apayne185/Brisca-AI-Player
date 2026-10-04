"""Turn a stored tournament into a Markdown leaderboard."""

from __future__ import annotations

import re
from pathlib import Path

from brisca.ratings import bootstrap_intervals, bradley_terry, wilson_interval
from brisca.store import AGENT_SUMMARY, PAIRWISE, ResultsStore

README_START = "<!-- leaderboard:start -->"
README_END = "<!-- leaderboard:end -->"


def leaderboard(
    store: ResultsStore, tournament: int, anchor: str | None = None, bootstrap: int = 500
) -> str:
    summary = store.query(AGENT_SUMMARY, tournament)
    agents = [row["agent"] for row in summary]
    outcomes = store.outcomes(tournament)
    ratings = bradley_terry(outcomes, agents, anchor)
    intervals = bootstrap_intervals(outcomes, agents, anchor, samples=bootstrap)
    games_per_pair = {(r["agent"], r["opponent"]): r for r in store.query(PAIRWISE, tournament)}
    order = sorted(agents, key=ratings.__getitem__, reverse=True)
    by_agent = {row["agent"]: row for row in summary}

    anchor_note = f", anchored at `{anchor}` = 0" if anchor else ""
    lines = [
        f"Bradley-Terry ratings on the Elo scale{anchor_note}, with 95% bootstrap intervals "
        "from resampling deals.",
        "",
        "| Rank | Agent | Elo | 95% CI | Score | Avg points | ms / move |",
        "| ---: | --- | ---: | :---: | ---: | ---: | ---: |",
    ]
    for rank, agent in enumerate(order, 1):
        row, (lo, hi) = by_agent[agent], intervals[agent]
        lines.append(
            f"| {rank} | `{agent}` | {ratings[agent]:+.0f} | [{lo:+.0f}, {hi:+.0f}] "
            f"| {row['score']:.3f} | {row['avg_points']:.1f} | {row['ms_per_move']:.2f} |"
        )

    lines += [
        "",
        "Head-to-head score of the row agent against the column agent "
        "(± half-width of the 95% Wilson interval):",
        "",
        "| | " + " | ".join(f"`{a}`" for a in order) + " |",
        "| --- |" + " :---: |" * len(order),
    ]
    for agent in order:
        cells = []
        for opponent in order:
            pair = games_per_pair.get((agent, opponent))
            if pair is None:
                cells.append("-")
                continue
            lo, hi = wilson_interval(pair["score"], pair["games"])
            cells.append(f"{pair['score']:.2f} ± {(hi - lo) / 2:.2f}")
        lines.append(f"| `{agent}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def update_readme(readme: Path, content: str) -> None:
    """Replace the text between the leaderboard markers in ``readme``."""
    text = readme.read_text()
    pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{readme} has no {README_START} ... {README_END} block")
    replacement = f"{README_START}\n{content}{README_END}"
    readme.write_text(pattern.sub(lambda _: replacement, text))
