import random

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from brisca import DECK
from brisca.encoding import NUM_ACTIONS, OBS_SIZE, action_mask, encode_observation
from brisca.engine import beats
from brisca.observation import observe
from tests.helpers import random_playout


@given(deal_seed=st.integers(0, 2**32 - 1), policy=st.randoms(use_true_random=False))
def test_encoding_is_well_formed(deal_seed: int, policy: random.Random) -> None:
    for state in random_playout(deal_seed, policy):
        obs = observe(state, state.to_play)
        x, mask = encode_observation(obs), action_mask(obs)

        assert x.shape == (OBS_SIZE,)
        assert x.dtype == np.float32
        assert ((x >= 0) & (x <= 1)).all()
        assert mask.shape == (NUM_ACTIONS,)
        assert [DECK[i] for i in np.flatnonzero(mask)] == sorted(obs.hand, key=lambda c: c.ordinal)

        hand, trick, played = x[:40], x[40:80], x[80:120]
        assert (hand + trick + played <= 1).all(), "a card is in at most one place"
        assert hand.sum() == len(obs.hand)
        assert played.sum() == 2 * len(obs.history)

        trump_plane, beats_plane = x[200:240], x[240:280]
        assert [DECK[i] for i in np.flatnonzero(trump_plane)] == [
            c for c in DECK if c.suit == obs.trump
        ]
        winners = [DECK[i] for i in np.flatnonzero(beats_plane)]
        if obs.current_trick:
            lead = obs.current_trick[0]
            assert winners == [c for c in DECK if beats(c, lead, obs.trump)]
        else:
            assert winners == []


def test_encoding_is_deterministic() -> None:
    obs = observe(next(random_playout(0, random.Random(0))), 0)
    assert np.array_equal(encode_observation(obs), encode_observation(obs))
