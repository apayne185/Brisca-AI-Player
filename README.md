# Brisca AI

[![CI](https://github.com/apayne185/Brisca-AI-Player/actions/workflows/ci.yml/badge.svg)](https://github.com/apayne185/Brisca-AI-Player/actions/workflows/ci.yml)
[![CodeQL](https://github.com/apayne185/Brisca-AI-Player/actions/workflows/codeql.yml/badge.svg)](https://github.com/apayne185/Brisca-AI-Player/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An imperfect-information game AI platform for **Brisca**, the Spanish trick-taking
card game. It covers the full ML lifecycle: a fast and correct simulator, search
and reinforcement-learning agents, statistically sound evaluation, tracked
experiments, an inference service, and a bot-detection model trained on
gameplay telemetry.

> **Status:** under active rebuild. See the [roadmap](docs/ROADMAP.md) for
> what is done and what is next.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/apayne185/Brisca-AI-Player.git
cd Brisca-AI-Player
make install   # environment, dev tools and git hooks
make check     # lint, strict type checking and tests, exactly as CI runs them
```

## The game

Brisca is played with the 40-card Spanish deck: four suits (oros, copas,
espadas, bastos) of ranks 1–7 and 10–12.

- Each player is dealt three cards. The next card is turned face up; its suit is
  **trump** for the whole game, and the card itself is the last one drawn.
- Players do not have to follow suit. A trick is won by the highest trump played
  or, if there is none, by the highest card of the suit that was led.
- The trick winner draws first and leads the next trick.
- Once every card is played, the player with more than 60 of the 120 points wins.

| Card | Ace (1) | Three (3) | King (12) | Knight (11) | Jack (10) | 7, 6, 5, 4, 2 |
| --- | --- | --- | --- | --- | --- | --- |
| Points | 11 | 10 | 4 | 3 | 2 | 0 |

Card strength follows the same order: 1 > 3 > 12 > 11 > 10 > 7 > 6 > 5 > 4 > 2.

As an AI problem it is partially observable, stochastic, sequential and
multi-agent: the opponent's hand and the deck order are hidden, so agents must
reason over *information sets* rather than single states.

## Project history

This started as a university project by Anna Payne and Daniel Rosel that
compared MCTS, alpha-beta, rule-based heuristics, a grid-searched
"hyper-heuristic" and an ensemble against a course server. That version is
preserved at the [`academic-v1`](https://github.com/apayne185/Brisca-AI-Player/tree/academic-v1) tag.

### Audit of v1

Reviewing the original code before the rebuild turned up defects that
invalidate its reported win rates (0.80–0.90):

| Defect | Consequence |
| --- | --- |
| Trick winner compared raw ranks only, ignoring trump, led suit and Brisca's card order | The local simulator did not play Brisca |
| Scores summed ranks instead of card points | Grid search and local evaluation optimised the wrong objective |
| MCTS compared an integer score to a player id, so it never recorded a win | MCTS chose moves effectively at random |
| Alpha-beta indexed a non-subscriptable state object at its depth limit | The search crashed instead of evaluating leaves |
| The ensemble checked votes before any were cast | Its consensus branch was dead code |
| Results came from 50 games against an external server, without seeds or intervals | Not reproducible; differences between agents were not significant |

The rebuild addresses these with an immutable, fully tested rules engine,
property-based tests for game invariants, seeded and reproducible evaluation,
and confidence intervals on every reported metric.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under [MIT](LICENSE).
