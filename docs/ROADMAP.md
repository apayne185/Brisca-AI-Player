# Roadmap

The goal is a small but production-grade ML system built around an
imperfect-information card game: correct simulation, a range of agents,
statistically sound evaluation, tracked experiments, a deployed inference
service, and a gameplay-telemetry bot-detection model.

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Foundations: package layout, tooling, CI/CD, branch protection, Dependabot | Done |
| 1 | Correct, fast, immutable game engine with information-set API and property tests | Done |
| 2 | Agent zoo behind one interface: random, greedy, heuristic, ISMCTS, determinized alpha-beta, PPO self-play (with a batched environment for training throughput) | Done |
| 3 | Evaluation: seeded duplicate-deal tournaments, Elo/TrueSkill with bootstrap CIs, results in DuckDB | Next |
| 4 | Training pipelines: MLflow tracking and registry with a significance-gated champion, Hydra configs | Planned |
| 5 | Bot detection: telemetry dataset, XGBoost classifier with grouped CV, calibration, SHAP, model card | Planned |
| 6 | Serving: FastAPI inference, Docker, playable web demo, Prometheus metrics, drift monitoring | Planned |
| 7 | Stretch: LLM move explanations and LLM-agent benchmark, Kafka event streaming, AWS IaC | Planned |

## Principles

- Every number in the README is reproducible from a config and a seed.
- Every reported metric carries a confidence interval.
- Each phase ships independently and leaves `main` green.
