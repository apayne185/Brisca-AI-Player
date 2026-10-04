"""Persist tournaments in DuckDB and analyse them with SQL."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb

from brisca.engine import NUM_TRICKS
from brisca.ratings import Outcome
from brisca.tournament import GameRecord, TournamentConfig

SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS tournament_ids;
CREATE TABLE IF NOT EXISTS tournaments (
    id         INTEGER PRIMARY KEY DEFAULT nextval('tournament_ids'),
    created_at TIMESTAMP DEFAULT current_timestamp,
    git_sha    VARCHAR,
    config     JSON
);
CREATE TABLE IF NOT EXISTS games (
    tournament_id INTEGER REFERENCES tournaments (id),
    agent0        VARCHAR NOT NULL,
    agent1        VARCHAR NOT NULL,
    deal_seed     BIGINT NOT NULL,
    score0        INTEGER NOT NULL,
    score1        INTEGER NOT NULL,
    winner        INTEGER,  -- NULL for a 60-60 draw
    seconds0      DOUBLE NOT NULL,
    seconds1      DOUBLE NOT NULL
);
-- Each game seen from both players' side, which makes per-agent queries simple.
CREATE OR REPLACE VIEW results AS
SELECT tournament_id, agent0 AS agent, agent1 AS opponent, deal_seed,
       CASE winner WHEN 0 THEN 1.0 WHEN 1 THEN 0.0 ELSE 0.5 END AS score,
       score0 AS points, seconds0 AS seconds
FROM games
UNION ALL
SELECT tournament_id, agent1, agent0, deal_seed,
       CASE winner WHEN 1 THEN 1.0 WHEN 0 THEN 0.0 ELSE 0.5 END,
       score1, seconds1
FROM games;
"""

AGENT_SUMMARY = f"""
SELECT agent,
       count(*)                                   AS games,
       avg(score)                                 AS score,
       avg(points)                                AS avg_points,
       1000 * sum(seconds) / (count(*) * {NUM_TRICKS}) AS ms_per_move
FROM results
WHERE tournament_id = $tournament
GROUP BY agent
ORDER BY score DESC
"""

PAIRWISE = """
SELECT agent, opponent, avg(score) AS score, count(*) AS games
FROM results
WHERE tournament_id = $tournament
GROUP BY agent, opponent
ORDER BY agent, opponent
"""


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class ResultsStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def save(self, config: TournamentConfig, records: Sequence[GameRecord]) -> int:
        row = self.con.execute(
            "INSERT INTO tournaments (git_sha, config) VALUES (?, ?) RETURNING id",
            [_git_sha(), json.dumps(dataclasses.asdict(config))],
        ).fetchone()
        assert row is not None
        tournament_id = int(row[0])
        if records:
            self.con.executemany(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(tournament_id, *dataclasses.astuple(r)) for r in records],
            )
        return tournament_id

    def latest_tournament(self) -> int:
        row = self.con.execute("SELECT max(id) FROM tournaments").fetchone()
        if row is None or row[0] is None:
            raise LookupError("no tournaments stored")
        return int(row[0])

    def query(self, sql: str, tournament: int) -> list[dict[str, Any]]:
        cursor = self.con.execute(sql, {"tournament": tournament})
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, values, strict=True)) for values in cursor.fetchall()]

    def export_games(self, tournament: int, path: str | Path) -> None:
        """Write one tournament's games to Parquet, for sharing raw results."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # COPY cannot take bound parameters, so the id is formatted as an int and
        # the path's quotes are escaped.
        target = str(path).replace("'", "''")
        self.con.execute(
            "COPY (SELECT * EXCLUDE (tournament_id) FROM games "
            f"WHERE tournament_id = {int(tournament)} ORDER BY agent0, agent1, deal_seed) "
            f"TO '{target}' (FORMAT parquet)"
        )

    def outcomes(self, tournament: int) -> list[Outcome]:
        rows = self.con.execute(
            """
            SELECT agent0, agent1,
                   CASE winner WHEN 0 THEN 1.0 WHEN 1 THEN 0.0 ELSE 0.5 END, deal_seed
            FROM games WHERE tournament_id = ?
            """,
            [tournament],
        ).fetchall()
        return [Outcome(a, b, float(s), int(d)) for a, b, s, d in rows]
