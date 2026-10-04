from pathlib import Path

import duckdb
import pytest

from brisca.cli import main
from brisca.report import README_END, README_START, leaderboard, update_readme
from brisca.store import AGENT_SUMMARY, PAIRWISE, ResultsStore
from brisca.tournament import AgentSpec, TournamentConfig, run_tournament

CONFIG = TournamentConfig(
    agents=(AgentSpec("random", "random", {"seed": 0}), AgentSpec("greedy", "greedy")),
    deals=5,
)


@pytest.fixture
def store() -> ResultsStore:
    store = ResultsStore()
    store.save(CONFIG, run_tournament(CONFIG, workers=1))
    return store


def test_saves_games_and_summarises_them_in_sql(store: ResultsStore) -> None:
    tournament = store.latest_tournament()
    summary = {row["agent"]: row for row in store.query(AGENT_SUMMARY, tournament)}
    assert summary["greedy"]["games"] == summary["random"]["games"] == 10
    assert summary["greedy"]["score"] + summary["random"]["score"] == pytest.approx(1.0)
    assert summary["greedy"]["avg_points"] + summary["random"]["avg_points"] == pytest.approx(120)

    pairwise = store.query(PAIRWISE, tournament)
    assert {(r["agent"], r["opponent"]) for r in pairwise} == {
        ("greedy", "random"),
        ("random", "greedy"),
    }
    assert len(store.outcomes(tournament)) == 10


def test_tournament_ids_increase(store: ResultsStore) -> None:
    first = store.latest_tournament()
    second = store.save(CONFIG, [])
    assert second == first + 1 == store.latest_tournament()


def test_empty_store_has_no_latest_tournament() -> None:
    with pytest.raises(LookupError):
        ResultsStore().latest_tournament()


def test_leaderboard_lists_agents_best_first(store: ResultsStore) -> None:
    text = leaderboard(store, store.latest_tournament(), anchor="random", bootstrap=20)
    assert "anchored at `random` = 0" in text
    assert text.index("| 1 | `greedy`") < text.index("| 2 | `random`")
    assert "| `greedy` | - |" in text


def test_update_readme_replaces_only_the_marked_block(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n")
    update_readme(readme, "new table\n")
    assert readme.read_text() == f"intro\n{README_START}\nnew table\n{README_END}\noutro\n"


def test_update_readme_requires_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers")
    with pytest.raises(ValueError, match="leaderboard:start"):
        update_readme(readme, "x")


def test_cli_runs_tournament_and_publishes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "t.toml"
    config.write_text(
        'deals = 2\n[[agents]]\nid = "random"\ntype = "random"\n'
        '[[agents]]\nid = "greedy"\ntype = "greedy"\n'
    )
    db = tmp_path / "results" / "r.duckdb"
    main(["tournament", str(config), "--db", str(db), "--workers", "1"])

    main(["report", "--db", str(db), "--bootstrap", "10"])
    assert "`greedy`" in capsys.readouterr().out

    out, readme = tmp_path / "board.md", tmp_path / "README.md"
    readme.write_text(f"{README_START}\n{README_END}\n")
    games = tmp_path / "games.parquet"
    publish = ["--out", str(out), "--readme", str(readme), "--export-games", str(games)]
    main(["report", "--db", str(db), "--bootstrap", "10", *publish])
    assert out.read_text() in readme.read_text()
    exported = duckdb.sql(f"SELECT count(*), sum(score0 + score1) FROM '{games}'").fetchone()
    assert exported == (4, 480)
