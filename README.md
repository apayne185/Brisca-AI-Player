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

## Using the engine

The rules engine is a set of pure functions over an immutable `GameState`, so
search algorithms can branch from any state without copying.

```python
import random
from brisca import determinize, legal_actions, new_game, observe, step, winner

rng = random.Random(0)
state = new_game(seed=rng)
while not state.is_terminal:
    obs = observe(state, state.to_play)  # what the player to move can see
    sample = determinize(obs, rng)  # one consistent guess at the hidden cards
    state = step(state, rng.choice(legal_actions(state)))

print(state.scores, winner(state))
```

| Module | Responsibility |
| --- | --- |
| `brisca.cards` | Spanish deck, card points, trick-taking strength, stable card ordinals for ML encodings |
| `brisca.engine` | `new_game`, `legal_actions`, `step`, `trick_winner`, `winner`, `returns` |
| `brisca.observation` | Per-player information sets (`observe`) and uniform sampling of consistent hidden states (`determinize`) |

The engine is covered by table-driven rule tests and Hypothesis property tests
that check invariants in every reachable state: each card exists exactly once,
scores equal the points of won tricks, the face-up trump is drawn last, the
trick winner leads, and every game is a 120-point zero-sum game. Observations
are tested for leaks: resampling the hidden cards never changes what a player
sees. A random game runs in about 145 µs on one core (`make bench`).

## Agents

Every agent implements one protocol, `act(observation) -> card`, and only ever
sees what its player could see at the table.

| Agent | Approach |
| --- | --- |
| `random` | Uniformly random legal card, the baseline floor |
| `greedy` | One-ply: best immediate point swing, cheapest card otherwise |
| `heuristic` | Rule-based tactics with tunable parameters: bank points when winning in suit, save trumps for valuable tricks, trump freely in the endgame |
| `ismcts` | Single-Observer Information Set MCTS: searches over the player's information set using determinized samples and availability-aware UCB |
| `alphabeta` | Perfect Information Monte Carlo: alpha-beta over sampled determinizations, and exact once the stock is empty |

```python
from brisca.agents import make_agent
from brisca.arena import play_match

result = play_match(make_agent("ismcts", iterations=500, seed=0), make_agent("heuristic"), deals=50)
print(result.score)  # win rate with draws as half, over 100 duplicate games
```

Matches use **duplicate deals**: each shuffled deal is played twice with the
seats swapped, which cancels most of the luck of the cards. Search agents are
tested against exact solvers: alpha-beta pruning must match plain minimax, and
both search agents must play solved endgames optimally. A seeded strength suite
in CI checks that every agent clearly beats random play. Ratings with confidence
intervals come in Phase 3.

## Reinforcement learning

A PPO agent learns by self-play (`pip install 'brisca[rl]'`, then `brisca-train`).
Each training episode is played against an opponent drawn from a league: the
random, greedy and heuristic agents plus frozen snapshots of the learner. The
policy is an action-masked actor-critic over the 40 cards.

```bash
brisca-train --total-steps 3000000 --out models/ppo.pt   # ~25 min on one CPU core
```

After 3M steps (about 150k games), scored over 1,000 duplicate games against
each opponent, with 95% confidence intervals:

| Opponent | Score, per-output policy head | Score, shared per-card head |
| --- | --- | --- |
| random | 0.873 ± 0.021 | 0.875 ± 0.021 |
| greedy | 0.534 ± 0.031 | 0.546 ± 0.031 |
| heuristic | 0.444 ± 0.031 | 0.442 ± 0.031 |

PPO comfortably beats random play and edges past the greedy agent, but it does
not yet beat the hand-tuned heuristic. Phase 4 adds systematic hyperparameter
search and experiment tracking to close that gap.

**What made it learn.** The first version stayed at random-level play however
it was tuned. Driving the environment with the greedy agent reproduced the
expected 0.86 score, which ruled out a reward or environment bug and pointed at
the input: with only one-hot card ids, the network had to rediscover which card
beats which from scratch. Adding two relational feature planes, *trump suit*
and *cards that beat the current trick*, took the agent from 0.52 to 0.74
against random within 100k steps. The shared per-card head (`--card-head`)
learns faster early on but converges to the same strength.

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
