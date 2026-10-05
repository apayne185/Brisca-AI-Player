# Model card: Brisca bot detector

Full evaluation: [bot-detection-report.md](bot-detection-report.md) (regenerated
by `brisca detect train`). Artefact: [`models/bot-detector/`](../models/bot-detector/).

## Summary

A gradient-boosted classifier (XGBoost) that scores each **player session** for
the likelihood that the player is automated. It ships with calibrated
probabilities and a threshold that flags at most **1% of human sessions**.

The shipped model uses **decision features only** (how the player plays, not
how fast). At the 1% false-positive budget it catches 87.6% of bot sessions in
the bot styles it was trained on and **86.2% of an unseen adaptive bot style**.
A model that also uses timing features scores 100% on known styles but only
16.7% on the adaptive one, so it was not shipped.

## Intended use

- **Use:** rank sessions for review by a trust and safety analyst, and
  prioritise the review queue by calibrated probability.
- **Do not use:** to ban or penalise players automatically. A score is
  evidence for a human to weigh, not a verdict. Decisions should combine
  several sessions and other signals (account age, payment, device and network
  data).

## Data

**All data is synthetic.** `brisca detect simulate` generates 2,000 players
(16% bots), 6,074 sessions and 549,200 moves:

- **Humans** have a skill level (Beta(2, 2)) and a personal tempo. They play
  the heuristic move with a probability that rises with skill and falls as
  they tire through a session, and otherwise play greedily or at random. They
  think longer when more points are at stake, play forced cards quickly, are
  occasionally distracted and take breaks between games.
- **Bots** play the heuristic, ISMCTS or `ppo-v2` agent, with one of four
  timing styles: `naive` (fast, flat), `jittered` (random human-scale delays),
  `humanized` (human-shaped delays keyed on hand size) and `mimic` (exactly the
  human timing model). `mimic` bots are **never used for training**; they stand
  in for an adversary that adapts after a detector launches.

Per-move telemetry includes annotations a real platform could compute from its
server-side game logs: agreement with known bot policies, and whether moves in
endgames (which are perfect-information and solved exactly) were optimal.

## Features

Session features are computed in SQL (DuckDB) from the raw tables; see
`brisca.detection.features`.

| Family | Features | Shipped |
| --- | --- | :---: |
| Decision | agreement with the heuristic and greedy policies, endgame accuracy, average points, win rate | yes |
| Timing | think-time level and spread, forced vs chosen move time, correlation of think time with stakes and hand size, long pauses, between-game gaps | no |

## Evaluation

- 5-fold cross-validation **grouped by player**, so no player is in both
  training and test data.
- Threshold chosen so at most 1% of human sessions are flagged; recall is
  reported at that operating point, the number that matters when false
  accusations are costly.
- Isotonic calibration fitted on players held out from the booster.

| Model | PR-AUC (seen styles) | Recall at 1% FPR (seen) | Recall, unseen `mimic` |
| --- | ---: | ---: | ---: |
| All features | 1.000 | 100.0% | 16.7% |
| Timing only | 0.997 | 99.6% | 1.8% |
| **Decision only (shipped)** | 0.947 | 87.6% | **86.2%** |
| Logistic regression, all features | 0.999 | 99.7% | 8.7% |

### Slices of the shipped model

| Group | Result |
| --- | ---: |
| Novice humans (skill < 0.33), wrongly flagged | 2.5% |
| Mid-skill humans, wrongly flagged | 0.5% |
| Expert humans (skill > 0.67), wrongly flagged | 0.3% |
| Bots running the heuristic, caught | 100.0% |
| Bots running ISMCTS, caught | 100.0% |
| Bots running `ppo-v2`, caught | 62.8% |

False positives are **not spread evenly**: novices are flagged about five
times as often as other humans. Their erratic play disagrees with the heuristic
in much the same way the PPO bot's play does, which is also why PPO bots are
the hardest to catch. Before production use this skew should be addressed, for
example with a per-segment threshold, or with features that separate erratic
human play from a consistent but unfamiliar policy.

## Limitations

- **Synthetic data.** The human model is a simplification; real players are
  more varied and real bots more inventive. Headline numbers are an upper
  bound for this design, not a forecast. The pipeline (telemetry tables → SQL
  features → grouped CV → calibrated model → slices) is built to take real
  telemetry instead.
- **Known-policy features.** Agreement with the heuristic and greedy agents
  works because some bots run those policies. Bots built on unknown policies
  (as the PPO slice shows) are harder to catch, and the feature set should
  track whatever bot frameworks are seen in the wild.
- **Session length.** Features are noisier for short sessions; scores from
  sessions with few endgames deserve less weight.
- **Adversarial drift.** The `mimic` result shows that any single signal can be
  learned and copied. Detection should be monitored and retrained as bots
  adapt; the service tracks feature drift (PSI against the training data) for
  exactly this reason.

## Ethical considerations

Flagging a real person as a bot can cost them their account and money. Hence:
a strict false-positive budget, calibrated scores that reviewers can reason
about, per-feature explanations (TreeSHAP) for every flagged session, slices by
player skill to detect unequal error rates, and a human in the loop for every
enforcement decision.
