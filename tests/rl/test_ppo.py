import math
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brisca.encoding import NUM_ACTIONS, OBS_SIZE, action_mask, encode_observation  # noqa: E402
from brisca.observation import observe  # noqa: E402
from brisca.rl import (  # noqa: E402
    ActorCritic,
    PolicyAgent,
    PPOConfig,
    load_checkpoint,
    save_checkpoint,
    train,
)
from brisca.rl.ppo import League, _gae  # noqa: E402
from brisca.rl.train import config_from, main, parse_args  # noqa: E402
from tests.helpers import random_playout  # noqa: E402

TINY = PPOConfig(
    total_steps=256,
    num_envs=4,
    rollout_len=32,
    epochs=1,
    minibatch_size=64,
    hidden=16,
    snapshot_every=1,
    eval_every=2,
    eval_deals=2,
)


@pytest.mark.parametrize("card_head", [False, True])
def test_model_never_puts_mass_on_illegal_cards(card_head: bool) -> None:
    model = ActorCritic(hidden=16, card_head=card_head)
    obs = torch.randn(8, OBS_SIZE)
    mask = torch.zeros(8, NUM_ACTIONS, dtype=torch.bool)
    mask[:, :3] = True
    logits, values = model(obs, mask)
    probs = torch.softmax(logits, dim=-1)
    assert torch.all(probs[:, 3:] == 0)
    assert torch.allclose(probs.sum(-1), torch.ones(8))
    assert values.shape == (8,)


@pytest.mark.parametrize("greedy", [True, False])
def test_policy_agent_plays_legal_cards(greedy: bool) -> None:
    agent = PolicyAgent(ActorCritic(hidden=16), greedy=greedy, seed=0)
    for state in random_playout(0, random.Random(0)):
        if not state.is_terminal:
            assert agent.act(observe(state, state.to_play)) in state.hands[state.to_play]


@pytest.mark.parametrize("card_head", [False, True])
def test_checkpoint_round_trip(tmp_path: Path, card_head: bool) -> None:
    model = ActorCritic(hidden=16, card_head=card_head)
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(model, path, {"note": "test"})
    loaded, metadata = load_checkpoint(path)
    assert metadata == {"note": "test"}

    obs = observe(next(random_playout(1, random.Random(1))), 0)
    x = torch.from_numpy(encode_observation(obs)).unsqueeze(0)
    m = torch.from_numpy(action_mask(obs)).unsqueeze(0)
    assert torch.equal(model(x, m)[0], loaded(x, m)[0])
    assert PolicyAgent.from_checkpoint(path).act(obs) in obs.hand


def test_gae_matches_hand_computation() -> None:
    # One env, three steps, episode ends at t=1 then a new one starts.
    rewards = torch.tensor([[0.0], [1.0], [0.5]])
    values = torch.tensor([[0.2], [0.4], [0.1]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last = torch.tensor([0.3])
    adv, ret = _gae(rewards, values, dones, last, gamma=1.0, lam=0.5)

    d2 = 0.5 + 0.3 - 0.1
    d1 = 1.0 - 0.4
    d0 = 0.0 + 0.4 - 0.2
    expected = torch.tensor([[d0 + 0.5 * d1], [d1], [d2]])
    assert torch.allclose(adv, expected)
    assert torch.allclose(ret, expected + values)


def test_league_mixes_baselines_and_snapshots() -> None:
    league = League(PPOConfig(self_play_prob=0.5, max_snapshots=2))
    rng = random.Random(0)
    assert {league.sample(rng).name for _ in range(50)} == {"random", "greedy", "heuristic"}
    for i in range(3):
        league.add_snapshot(ActorCritic(hidden=16), seed=i)
    assert len(league.snapshots) == 2
    assert "ppo" in {league.sample(rng).name for _ in range(50)}


def test_training_reports_finite_metrics() -> None:
    reports: list[tuple[int, dict[str, float]]] = []
    model = train(TINY, on_metrics=lambda step, m: reports.append((step, m)))
    assert isinstance(model, ActorCritic)
    assert [step for step, _ in reports] == [128, 256]
    assert "eval_vs_heuristic" in reports[-1][1]
    assert all(math.isfinite(v) for _, m in reports for v in m.values())


def test_training_is_reproducible() -> None:
    first, second = train(TINY), train(TINY)
    for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_cli_trains_and_saves(tmp_path: Path) -> None:
    out = tmp_path / "ppo.pt"
    main(["--total-steps", "128", "--num-envs", "4", "--rollout-len", "32", "--hidden", "16",
          "--eval-every", "1", "--eval-deals", "1", "--out", str(out), "--no-mlflow"])  # fmt: skip
    _, metadata = load_checkpoint(out)
    assert metadata["config"]["total_steps"] == 128


def test_cli_exposes_every_config_field() -> None:
    args = parse_args(["--learning-rate", "0.002", "--shaping", "0", "--card-head"])
    config = config_from(args)
    assert config.learning_rate == 0.002
    assert config.shaping == 0.0
    assert config.card_head is True
    assert config_from(parse_args(["--no-card-head"])).card_head is False
    assert args.out == Path("models/ppo.pt")


def test_cli_layers_flags_over_toml_config(tmp_path: Path) -> None:
    path = tmp_path / "ppo.toml"
    path.write_text("learning_rate = 0.0005\nhidden = 64\ncard_head = true\n")
    config = config_from(parse_args(["--config", str(path), "--hidden", "32"]))
    assert config.learning_rate == 0.0005
    assert config.card_head is True
    assert config.hidden == 32


def test_cli_rejects_unknown_toml_settings(tmp_path: Path) -> None:
    path = tmp_path / "ppo.toml"
    path.write_text("learning_rat = 0.1\n")
    with pytest.raises(SystemExit, match="learning_rat"):
        parse_args(["--config", str(path)])


@pytest.mark.slow
def test_ppo_learns_to_beat_random() -> None:
    from brisca.agents import RandomAgent
    from brisca.rl.ppo import evaluate

    config = PPOConfig(total_steps=100_000, eval_every=10**9)
    assert evaluate(train(config), RandomAgent(seed=0), deals=100) >= 0.65
