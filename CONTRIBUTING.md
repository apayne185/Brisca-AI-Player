# Contributing

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install   # creates .venv, installs dev tools, installs git hooks
make check     # lint + typecheck + tests with coverage, the same as CI
```

Run `make` to list every task.

## Workflow

`main` is protected: every change lands through a pull request that is
squash-merged once CI is green.

1. Branch from `main`: `git switch -c feat/short-description`.
2. Commit in small, focused steps. Pre-commit hooks run ruff, mypy and
   other checks on every commit.
3. Open a PR whose title follows [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `ci:`, `build:`, `chore:`, `perf:`).
   The title becomes the commit message on `main`.

## Standards

- **Types**: the package is checked with `mypy --strict`.
- **Tests**: new behaviour needs tests; coverage must stay at or above 90%.
  Prefer property-based tests (`hypothesis`) for game-rule invariants.
- **Reproducibility**: anything stochastic takes an explicit seed or generator
  (`random.Random` in the engine). No global random state.
- **Results**: any reported metric states the number of games, the seeds and a
  confidence interval.

## Releases

Bump `version` in `pyproject.toml` through a PR, then push a matching tag
(`git tag v0.2.0 && git push origin v0.2.0`). The release workflow builds the
package, attests its provenance and publishes a GitHub release.
