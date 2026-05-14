from enum import Enum
from dataclasses import asdict, dataclass


class Street(Enum):
    PREFLOP = "Preflop"
    FLOP = "Flop"
    TURN = "Turn"
    RIVER = "River"

    def __str__(self):
        return self.value


@dataclass
class PlayerState:
    index: int
    stack: int
    hole_cards: list[dict]
    folded: bool
    contributed_street: int


@dataclass
class PokerState:
    players: list[PlayerState]
    board: list[dict]
    pot: int
    street: Street
    dealer: int
    acting_idx: int | None
    current_bet: int
    min_raise: int
    hand_over: bool
    winner: int | None


@dataclass
class Action:
    player: int
    street: Street
    type: int  # 0: "fold", 1: "call", 2: "raise"
    amount: int  # total put in (or raise size)


@dataclass
class PokerState:
    players: list[PlayerState]
    board: list[str]
    pot: int
    street: Street
    dealer: int
    acting_idx: int | None
    current_bet: int
    min_raise: int
    hand_over: bool
    winner: int | None
    action_history: list[Action]
