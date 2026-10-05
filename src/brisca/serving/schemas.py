"""Request and response models. Cards are compact strings such as ``"3O"`` or ``"12E"``."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from brisca.cards import Card
from brisca.engine import Trick
from brisca.observation import Observation


def _valid_card(text: str) -> str:
    Card.parse(text)
    return text


CardStr = Annotated[str, AfterValidator(_valid_card), Field(examples=["3O"])]


class TrickIn(BaseModel):
    leader: int = Field(ge=0, le=1)
    cards: list[CardStr] = Field(min_length=2, max_length=2)
    winner: int = Field(ge=0, le=1)


class ObservationIn(BaseModel):
    """What one player can see; mirrors ``brisca.observation.Observation``."""

    player: int = Field(ge=0, le=1)
    hand: list[CardStr] = Field(min_length=1, max_length=3)
    trump_card: CardStr
    current_trick: list[CardStr] = Field(default_factory=list, max_length=1)
    to_play: int = Field(ge=0, le=1)
    scores: tuple[int, int] = (0, 0)
    history: list[TrickIn] = Field(default_factory=list, max_length=20)
    stock_size: int = Field(ge=0, le=34)
    opponent_hand_size: int = Field(ge=0, le=3)

    def to_observation(self) -> Observation:
        def cards(texts: list[str]) -> tuple[Card, ...]:
            return tuple(Card.parse(t) for t in texts)

        return Observation(
            player=self.player,
            hand=cards(self.hand),
            trump_card=Card.parse(self.trump_card),
            current_trick=cards(self.current_trick),
            to_play=self.to_play,
            scores=self.scores,
            history=tuple(Trick(t.leader, cards(t.cards), t.winner) for t in self.history),
            stock_size=self.stock_size,
            opponent_hand_size=self.opponent_hand_size,
        )


class MoveRequest(BaseModel):
    agent: str = Field(examples=["heuristic"])
    observation: ObservationIn


class MoveResponse(BaseModel):
    agent: str
    card: CardStr
    latency_ms: float


class AgentInfo(BaseModel):
    id: str
    type: str
    params: dict[str, object]


class BotScoreRequest(BaseModel):
    """Session features as produced by ``brisca.detection.features`` (missing values allowed)."""

    features: dict[str, float | None] = Field(
        examples=[
            {
                "agree_heuristic": 0.95,
                "agree_greedy": 0.6,
                "endgame_accuracy": 1.0,
                "avg_points": 66.0,
                "win_rate": 0.75,
            }
        ]
    )


class Contribution(BaseModel):
    feature: str
    value: float | None
    contribution: float
    """SHAP value in log-odds; positive pushes towards 'bot'."""


class BotScoreResponse(BaseModel):
    probability: float
    flagged: bool
    contributions: list[Contribution]


class NewGameRequest(BaseModel):
    agent: str = "heuristic"
    seed: int | None = None


class TrickView(BaseModel):
    leader: str
    cards: list[CardStr]
    winner: str
    points: int


class GameView(BaseModel):
    """The human's view of a demo game (the human is always player 0)."""

    game_id: str
    agent: str
    hand: list[CardStr]
    trump_card: CardStr
    trump_drawn: bool
    current_trick: list[CardStr]
    your_turn: bool
    scores: dict[str, int]
    stock_size: int
    last_trick: TrickView | None
    finished: bool
    result: str | None


class MoveIn(BaseModel):
    card: CardStr
