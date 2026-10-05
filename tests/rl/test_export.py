import random
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from brisca.observation import observe
from brisca.onnx_policy import OnnxPolicyAgent
from brisca.rl import ActorCritic, PolicyAgent, save_checkpoint
from brisca.rl.export import export_onnx, main
from brisca.tournament import AgentSpec, build_agent
from tests.helpers import random_playout


@pytest.mark.parametrize("card_head", [False, True])
def test_onnx_agent_matches_pytorch(tmp_path: Path, card_head: bool) -> None:
    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(ActorCritic(hidden=32, card_head=card_head), checkpoint, {})
    path = export_onnx(checkpoint, tmp_path / "policy.onnx")  # includes a parity check

    torch_agent = PolicyAgent.from_checkpoint(checkpoint)
    onnx_agent = OnnxPolicyAgent(path)
    for state in random_playout(2, random.Random(2)):
        if not state.is_terminal:
            obs = observe(state, state.to_play)
            assert onnx_agent.act(obs) == torch_agent.act(obs)


def test_export_cli_and_tournament_type(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(ActorCritic(hidden=16), checkpoint, {})
    main([str(checkpoint), str(tmp_path / "out.onnx")])
    assert "parity check passed" in capsys.readouterr().out
    agent = build_agent(AgentSpec("p", "onnx", {"path": str(tmp_path / "out.onnx")}))
    assert agent.name == "ppo"
