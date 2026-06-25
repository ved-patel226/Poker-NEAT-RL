export type CardVector = [number, number];

export type TracePlayer = {
    index: number;
    stack: number;
    hole_cards: CardVector[];
    folded: boolean;
    contributed_street: number;
};

export type TraceState = {
    players: TracePlayer[];
    board: CardVector[];
    pot: number;
    acting_idx: number | null;
    street: string;
    dealer: number;
    current_bet: number;
    min_raise: number;
    hand_over: boolean;
    winner: number | null;
    action_history?: unknown[];
};



export type TraceAction = {
    player: number;
    street: string;
    type: number;
    amount: number;
};


export type TraceEvent = {
    player: number;
    state: TraceState;
    action: TraceAction;
};


export type ActionBounds = {
    action_type_space: number;
    can_fold: boolean;
    can_call: boolean;
    can_raise: boolean;
    can_all_in: boolean;
    raise_amount_min: number;
    raise_amount_max: number;
}