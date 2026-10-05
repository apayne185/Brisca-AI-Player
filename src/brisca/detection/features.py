"""Session-level features computed in SQL from raw telemetry.

One row per player session. Features fall in two families, which the
evaluation ablates separately:

* **timing**: how long moves take and, crucially, whether thinking time tracks
  how much a decision matters. Humans slow down when the stakes are high;
  scripted delays usually don't.
* **decision**: what is played: agreement with known bot policies, accuracy
  in endgames that the server can solve exactly, and results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import numpy.typing as npt

SESSION_FEATURES_SQL = """
WITH moves_agg AS (
    SELECT player_id, session_id,
        avg(ln(think_s))                                         AS mean_log_think,
        stddev_samp(ln(think_s))                                 AS sd_log_think,
        quantile_cont(think_s, 0.1)                              AS p10_think,
        quantile_cont(think_s, 0.9)                              AS p90_think,
        median(think_s) FILTER (WHERE n_options = 1)             AS forced_think,
        median(think_s) FILTER (WHERE n_options > 1)             AS choice_think,
        corr(ln(think_s), stakes) FILTER (WHERE n_options > 1)   AS think_stakes_corr,
        corr(ln(think_s), n_options)                             AS think_options_corr,
        avg(CASE WHEN think_s > 10 THEN 1.0 ELSE 0.0 END)        AS long_pause_rate,
        avg(agrees_heuristic::DOUBLE)                            AS agree_heuristic,
        avg(agrees_greedy::DOUBLE)                               AS agree_greedy,
        avg(endgame_optimal::DOUBLE) FILTER (WHERE endgame_decision) AS endgame_accuracy
    FROM moves
    GROUP BY player_id, session_id
),
gaps AS (
    SELECT player_id, session_id, points, won,
        start_ts - lag(end_ts) OVER (
            PARTITION BY player_id, session_id ORDER BY game_id
        ) AS gap
    FROM games
),
games_agg AS (
    SELECT player_id, session_id,
        avg(points)             AS avg_points,
        avg(won::DOUBLE)        AS win_rate,
        avg(ln(gap))            AS mean_log_gap,
        stddev_samp(ln(gap))    AS sd_log_gap
    FROM gaps
    GROUP BY player_id, session_id
)
SELECT
    m.*,
    g.* EXCLUDE (player_id, session_id),
    m.choice_think / m.forced_think   AS choice_forced_ratio,
    p.is_bot::INTEGER                 AS label,
    coalesce(p.bot_style, 'human')    AS bot_style,
    p.policy,
    p.skill
FROM moves_agg m
JOIN games_agg g USING (player_id, session_id)
JOIN players p USING (player_id)
ORDER BY player_id, session_id
"""

TIMING_FEATURES = (
    "mean_log_think",
    "sd_log_think",
    "p10_think",
    "p90_think",
    "forced_think",
    "choice_think",
    "choice_forced_ratio",
    "think_stakes_corr",
    "think_options_corr",
    "long_pause_rate",
    "mean_log_gap",
    "sd_log_gap",
)
DECISION_FEATURES = (
    "agree_heuristic",
    "agree_greedy",
    "endgame_accuracy",
    "avg_points",
    "win_rate",
)
ALL_FEATURES = TIMING_FEATURES + DECISION_FEATURES

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Dataset:
    features: tuple[str, ...]
    X: FloatArray
    y: npt.NDArray[np.int64]
    groups: npt.NDArray[np.int64]
    """Player id per row: every split must keep a player's sessions together."""
    bot_style: npt.NDArray[np.str_]
    policy: npt.NDArray[np.str_]
    skill: FloatArray

    def select(self, features: tuple[str, ...]) -> FloatArray:
        return self.X[:, [self.features.index(f) for f in features]]


def _column(values: npt.NDArray[np.generic]) -> FloatArray:
    masked = np.ma.masked_invalid(np.ma.asarray(values, dtype=np.float64))
    return np.asarray(masked.filled(np.nan), dtype=np.float64)


def load_dataset(db: str | Path) -> Dataset:
    with duckdb.connect(str(db), read_only=True) as con:
        data = con.execute(SESSION_FEATURES_SQL).fetchnumpy()
    return Dataset(
        features=ALL_FEATURES,
        X=np.column_stack([_column(data[f]) for f in ALL_FEATURES]),
        y=np.asarray(data["label"], dtype=np.int64),
        groups=np.asarray(data["player_id"], dtype=np.int64),
        bot_style=np.asarray(data["bot_style"], dtype=np.str_),
        policy=np.asarray(data["policy"], dtype=np.str_),
        skill=_column(data["skill"]),
    )
