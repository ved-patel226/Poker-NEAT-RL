try:
    from .states import Street, PlayerState, PokerState, Action
except ImportError:
    from states import Street, PlayerState, PokerState, Action


from pokerkit import NoLimitTexasHoldem, Mode, Automation
import torch

# (2-A) is 13-dim
RANK_ORDER = [
    "DEUCE",
    "TREY",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "JACK",
    "QUEEN",
    "KING",
    "ACE",
]
SUIT_ORDER = ["CLUB", "DIAMOND", "HEART", "SPADE"]


def make_card_vec(card) -> list[float]:
    rank = RANK_ORDER.index(card.rank.name) / 12.0
    suit = SUIT_ORDER.index(card.suit.name) / 3.0
    return [rank, suit]


class Observation:
    def __init__(self, config=None):
        if config:
            print("Config isn't supported right now...")

        self.starting_stacks = (10000, 10000, 10000, 10000, 10000, 10000)
        self.nominal_stack = self.starting_stacks[0]

        self.reset()

    def reset(self):
        self.state = NoLimitTexasHoldem.create_state(
            (
                Automation.ANTE_POSTING,  # automate everything... why is doing ts manual even an option??
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING,
                Automation.HOLE_DEALING,
                Automation.BOARD_DEALING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            True,  # uniform ante
            {-1: 600},  # 600 ante
            (200, 400, 800),
            400,  # min-bet
            self.starting_stacks,  # starting stacks
            6,  # number of players
            mode=Mode.CASH_GAME,
        )

        return self.state

    def get_state(self) -> dict:
        from dataclasses import asdict

        street_map = {
            0: Street.PREFLOP,
            1: Street.FLOP,
            2: Street.TURN,
            3: Street.RIVER,
        }

        players = []
        for i in range(self.state.player_count):
            hole_c = self.state.hole_cards
            hc_dicts = []
            if i < len(hole_c) and hole_c[i]:
                hc_dicts = [make_card_vec(c) for c in hole_c[i]]

            folded = not self.state.statuses[i]
            contributed = self.state.bets[i] if self.state.bets else 0

            players.append(
                PlayerState(
                    index=i,
                    stack=self.state.stacks[i],
                    hole_cards=hc_dicts,
                    folded=folded,
                    contributed_street=contributed,
                )
            )

        board_c = self.state.board_cards

        flat_board = []
        for item in board_c:
            try:  # extend if list/dict
                flat_board.extend(item)
            except TypeError:  # otherwise js append
                flat_board.append(item)

        board = [make_card_vec(c) for c in flat_board]

        acting_idx = self.state.actor_index

        dealer = 0

        current_bet = max(self.state.bets) if self.state.bets else 0
        min_raise = self.state.min_completion_betting_or_raising_to_amount
        if min_raise is None:
            min_raise = 0
        hand_over = not self.state.status
        pot = self.state.total_pot_amount

        winner = None
        if hand_over:
            payoffs = self.state.payoffs
            for i, p in enumerate(payoffs):
                if p > 0:
                    winner = i
                    break

        poker_state = PokerState(
            players=players,
            board=board,
            pot=pot,
            street=street_map.get(self.state.street_index, Street.PREFLOP),
            dealer=dealer,
            acting_idx=acting_idx,
            current_bet=current_bet,
            min_raise=min_raise,
            hand_over=hand_over,
            winner=winner,
            action_history=[],
        )

        return asdict(poker_state)

    def send_action(self, action: Action) -> None:
        if self.state.actor_index != action.player:
            raise ValueError(
                f"{action.player} acted out of turn. It is player {self.state.actor_index}'s turn."
            )

        if action.type == 0:  # fold
            if self.state.can_fold():
                self.state.fold()
            else:
                raise ValueError("Cannot fold in current state.")

        elif action.type == 1:  # call / check
            if self.state.can_check_or_call():
                self.state.check_or_call()
            else:
                raise ValueError("Cannot check or call in current state.")

        elif action.type == 2:  # raise
            if self.state.can_complete_bet_or_raise_to(action.amount):
                self.state.complete_bet_or_raise_to(action.amount)
            else:
                raise ValueError(
                    f"Cannot complete bet or raise to {action.amount} in current state."
                )

        else:
            raise ValueError(f"Invalid action type {action.type}. Expected 0, 1, or 2.")

    def get_tensor_input(self, player_idx: int = None) -> torch.Tensor:
        state = self.get_state()

        x = torch.zeros(39, dtype=torch.float32)
        scalar_scale = float(max(self.nominal_stack, 1))

        ptr = 0

        # GLOBAL
        x[ptr] = float(state["pot"]) / scalar_scale
        ptr += 1

        x[ptr] = float(state["current_bet"]) / scalar_scale
        ptr += 1

        x[ptr] = float(state["min_raise"]) / scalar_scale
        ptr += 1

        street = state["street"].value
        if street == "Preflop":
            x[ptr] = 1.0
        elif street == "Flop":
            x[ptr + 1] = 1.0
        elif street == "Turn":
            x[ptr + 2] = 1.0
        elif street == "River":
            x[ptr + 3] = 1.0
        ptr += 4

        # BOARD
        board = state["board"]

        for i in range(5):
            if i < len(board):
                vec = board[i]
                x[ptr] = vec[0]
                x[ptr + 1] = vec[1]
            ptr += 2

        # SELF
        me = state["players"][player_idx]

        x[ptr] = me["stack"] / scalar_scale
        ptr += 1

        x[ptr] = float(me["folded"])
        ptr += 1

        x[ptr] = me["contributed_street"] / scalar_scale
        ptr += 1

        hole_cards = me["hole_cards"]

        for i in range(2):
            if i < len(hole_cards):
                vec = hole_cards[i]
                x[ptr] = vec[0]
                x[ptr + 1] = vec[1]
            ptr += 2

        # OPPONENTS (public only info ofc)
        for p in state["players"]:
            if p["index"] == player_idx:
                continue

            x[ptr] = p["stack"] / scalar_scale
            ptr += 1

            x[ptr] = float(p["folded"])
            ptr += 1

            x[ptr] = p["contributed_street"] / scalar_scale
            ptr += 1

        return x

    def get_action_bounds(self) -> dict:
        min_amount = self.state.min_completion_betting_or_raising_to_amount
        actor_idx = self.state.actor_index

        max_amount = 0
        if actor_idx is not None:
            current_bet = (
                self.state.bets[actor_idx]
                if self.state.bets and actor_idx < len(self.state.bets)
                else 0
            )
            max_amount = self.state.stacks[actor_idx] + current_bet

        can_fold = self.state.can_fold()
        can_call = self.state.can_check_or_call()
        can_raise = min_amount is not None and self.state.can_complete_bet_or_raise_to(
            min_amount
        )
        can_all_in = (
            self.state.can_complete_bet_or_raise_to(max_amount)
            if max_amount > 0
            else False
        )

        return {
            "action_type_space": 3,
            "can_fold": can_fold,
            "can_call": can_call,
            "can_raise": can_raise,
            "can_all_in": can_all_in,
            "raise_amount_min": float(min_amount) if min_amount is not None else 0.0,
            "raise_amount_max": float(max_amount),
        }


def main() -> None:
    obs = Observation()
    state_dict = obs.get_tensor_input()
    action_bounds = obs.get_action_bounds()

    print(state_dict)
    print(action_bounds)


if __name__ == "__main__":
    main()
