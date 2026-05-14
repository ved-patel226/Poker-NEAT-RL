try:
    from .states import Street, PlayerState, PokerState, Action
except ImportError:
    from states import Street, PlayerState, PokerState, Action


from pokerkit import NoLimitTexasHoldem, Mode, Automation

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


def make_card_dict(card) -> dict:
    r_name = card.rank.name
    s_name = card.suit.name

    rank_oh = [0] * 13
    if r_name in RANK_ORDER:
        rank_oh[RANK_ORDER.index(r_name)] = 1

    suit_oh = [0] * 4
    if s_name in SUIT_ORDER:
        suit_oh[SUIT_ORDER.index(s_name)] = 1

    return {"rank_one_hot": rank_oh, "suit_one_hot": suit_oh}


class Observation:
    def __init__(self, config=None):
        if config:
            print("Config isn't supported right now...")

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
            (10000, 10000, 10000, 10000, 10000, 10000),  # starting stacks
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
                hc_dicts = [make_card_dict(c) for c in hole_c[i]]

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

        board = [make_card_dict(c) for c in flat_board]

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
                f"Player {action.player} acted out of turn. It is player {self.state.actor_index}'s turn."
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

    def get_tensor_input(self, player_idx: int = None):
        """
        Returns a flat PyTorch tensor representing the current state.
        Optionally masks hole cards for other players if `player_idx` is specified (Perfect for RL agents).
        """
        import torch

        state_dict = self.get_state()

        # global stats
        pot = float(state_dict["pot"])
        curr_bet = float(state_dict["current_bet"])
        min_raise = float(state_dict["min_raise"])

        # street one-hot (PREFLOP, FLOP, TURN, RIVER)
        street_names = ["Preflop", "Flop", "Turn", "River"]
        street_idx = (
            street_names.index(state_dict["street"].value)
            if state_dict["street"].value in street_names
            else 0
        )
        street_oh = [0.0] * 4
        street_oh[street_idx] = 1.0

        # board (up to 5 cards, 13+4=17 dims each) -> 85 dims
        board_features = []
        for i in range(5):
            if i < len(state_dict["board"]):
                c = state_dict["board"][i]
                board_features.extend(c["rank_one_hot"] + c["suit_one_hot"])
            else:
                board_features.extend([0.0] * 17)

        # players (6 players * (stack + folded + contributed + 2*17)) -> 6 * 37 = 222 dims
        player_features = []
        for p in state_dict["players"]:
            player_features.append(float(p["stack"]))
            player_features.append(float(p["folded"]))
            player_features.append(float(p["contributed_street"]))

            # hole cards (up to 2 cards) -> 34 dims per player
            for i in range(2):
                # mask other players' cards if player_idx is requested
                if i < len(p["hole_cards"]) and (
                    player_idx is None or player_idx == p["index"]
                ):
                    c = p["hole_cards"][i]
                    player_features.extend(c["rank_one_hot"] + c["suit_one_hot"])
                else:
                    player_features.extend([0.0] * 17)

        # feature vector size: 3 + 4 + 85 + 222 = 314 dimensions
        all_features = (
            [pot, curr_bet, min_raise] + street_oh + board_features + player_features
        )
        return torch.tensor(all_features, dtype=torch.float32)

    def get_action_bounds(self) -> dict:
        """
        Returns the action bounds for the currently acting player for the model prediction output.
        """
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
