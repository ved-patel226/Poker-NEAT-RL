import type { ActionBounds, TraceState } from "../components/Poker/pokerTypes";
import { bestHandScore, compareScores, type Card } from "./handEvaluator";

const NUM_PLAYERS = 6;
const STARTING_STACK = 10000;
const SMALL_BLIND = 200;
const BIG_BLIND = 400;
const MIN_BET = 400;

const PREFLOP_ORDER = [2, 3, 4, 5, 0, 1];
const POSTFLOP_ORDER = [0, 1, 2, 3, 4, 5];
const STREETS = ["Preflop", "Flop", "Turn", "River"];

type PlayerInternal = {
    stack: number;
    holeCards: Card[];
    folded: boolean;
    contributedStreet: number;
    contributedTotal: number;
    allIn: boolean;
};

function toCardVector(card: Card): [number, number] {
    return [(card.rank - 2) / 12, card.suit / 3];
}

function makeDeck(): Card[] {
    const deck: Card[] = [];
    for (let suit = 0; suit < 4; suit++) {
        for (let rank = 2; rank <= 14; rank++) deck.push({ rank, suit });
    }
    return deck;
}

function shuffle<T>(arr: T[]): T[] {
    const copy = [...arr];
    for (let i = copy.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
}

export class PokerEngine {
    private players: PlayerInternal[] = [];
    private deck: Card[] = [];
    private board: Card[] = [];
    private streetIndex = 0;
    private actingIdx: number | null = null;
    private currentBet = 0;
    private lastRaiseSize = MIN_BET;
    private handOver = false;
    private winner: number | null = null;
    private needsToAct = new Set<number>();

    constructor() {
        this.reset();
    }

    reset(): void {
        this.deck = shuffle(makeDeck());
        this.board = [];
        this.streetIndex = 0;
        this.handOver = false;
        this.winner = null;

        this.players = Array.from({ length: NUM_PLAYERS }, () => ({
            stack: STARTING_STACK,
            holeCards: [] as Card[],
            folded: false,
            contributedStreet: 0,
            contributedTotal: 0,
            allIn: false,
        }));

        for (let round = 0; round < 2; round++) {
            for (let i = 0; i < NUM_PLAYERS; i++) {
                this.players[i].holeCards.push(this.deck.pop()!);
            }
        }

        this.postBlind(0, SMALL_BLIND);
        this.postBlind(1, BIG_BLIND);
        this.currentBet = BIG_BLIND;
        this.lastRaiseSize = MIN_BET;

        this.beginBettingRound();
    }

    private postBlind(idx: number, amount: number) {
        const p = this.players[idx];
        const actual = Math.min(amount, p.stack);
        p.stack -= actual;
        p.contributedStreet += actual;
        p.contributedTotal += actual;
        if (p.stack === 0) p.allIn = true;
    }

    private actionOrderForStreet(): number[] {
        return this.streetIndex === 0 ? PREFLOP_ORDER : POSTFLOP_ORDER;
    }

    private activePlayers(): number[] {
        return this.players.map((_, i) => i).filter((i) => !this.players[i].folded);
    }

    private playersWhoCanAct(): number[] {
        return this.players
            .map((_, i) => i)
            .filter((i) => !this.players[i].folded && !this.players[i].allIn);
    }

    private beginBettingRound() {
        if (this.activePlayers().length <= 1) {
            this.finishHandUncontested();
            return;
        }

        const canAct = this.playersWhoCanAct();
        if (canAct.length <= 1) {
            this.runOutAndShowdown();
            return;
        }

        this.needsToAct = new Set(canAct);
        const order = this.actionOrderForStreet();
        this.actingIdx = order.find((i) => this.needsToAct.has(i)) ?? null;
    }

    private drawCards(n: number): Card[] {
        const cards: Card[] = [];
        for (let i = 0; i < n; i++) cards.push(this.deck.pop()!);
        return cards;
    }

    private runOutAndShowdown() {
        while (this.streetIndex < 3) {
            this.streetIndex += 1;
            this.board.push(...this.drawCards(this.streetIndex === 1 ? 3 : 1));
        }
        this.showdown();
    }

    private advanceStreet() {
        this.streetIndex += 1;
        if (this.streetIndex > 3) {
            this.showdown();
            return;
        }

        this.players.forEach((p) => (p.contributedStreet = 0));
        this.currentBet = 0;
        this.lastRaiseSize = MIN_BET;
        this.board.push(...this.drawCards(this.streetIndex === 1 ? 3 : 1));

        this.beginBettingRound();
    }

    private advanceTurn() {
        if (this.activePlayers().length <= 1) {
            this.finishHandUncontested();
            return;
        }

        if (this.needsToAct.size === 0) {
            this.advanceStreet();
            return;
        }

        const order = this.actionOrderForStreet();
        const startPos = order.indexOf(this.actingIdx!);
        for (let step = 1; step <= order.length; step++) {
            const idx = order[(startPos + step) % order.length];
            if (this.needsToAct.has(idx)) {
                this.actingIdx = idx;
                return;
            }
        }

        this.advanceStreet();
    }

    private awardPot(winnerIdx: number) {
        const pot = this.players.reduce((sum, p) => sum + p.contributedTotal, 0);
        this.players[winnerIdx].stack += pot;
    }

    private finishHandUncontested() {
        this.handOver = true;
        const remaining = this.activePlayers();
        this.winner = remaining[0] ?? null;
        if (this.winner !== null) this.awardPot(this.winner);
        this.actingIdx = null;
    }

    private showdown() {
        this.handOver = true;
        const contenders = this.activePlayers();

        let bestIdx = contenders[0];
        let bestScore = bestHandScore([...this.players[bestIdx].holeCards, ...this.board]);

        for (const idx of contenders.slice(1)) {
            const score = bestHandScore([...this.players[idx].holeCards, ...this.board]);
            if (compareScores(score, bestScore) > 0) {
                bestScore = score;
                bestIdx = idx;
            }
        }

        this.winner = bestIdx;
        this.awardPot(bestIdx);
        this.actingIdx = null;
    }

    sendAction(action: { player: number; type: number; amount: number }): void {
        if (this.handOver || this.actingIdx === null) {
            throw new Error("No action expected right now");
        }
        if (action.player !== this.actingIdx) {
            throw new Error(`It is not player ${action.player}'s turn`);
        }

        const p = this.players[action.player];
        const toCall = this.currentBet - p.contributedStreet;

        if (action.type === 0) {
            p.folded = true;
            this.needsToAct.delete(action.player);
        } else if (action.type === 1) {
            const callAmount = Math.min(toCall, p.stack);
            p.stack -= callAmount;
            p.contributedStreet += callAmount;
            p.contributedTotal += callAmount;
            if (p.stack === 0) p.allIn = true;
            this.needsToAct.delete(action.player);
        } else if (action.type === 2) {
            const minRaiseTo = this.currentBet + this.lastRaiseSize;
            const maxRaiseTo = p.stack + p.contributedStreet;
            const floor = Math.min(minRaiseTo, maxRaiseTo);

            if (action.amount < floor || action.amount > maxRaiseTo) {
                throw new Error(
                    `Raise amount ${action.amount} out of bounds [${floor}, ${maxRaiseTo}]`,
                );
            }

            const additional = action.amount - p.contributedStreet;
            p.stack -= additional;
            p.contributedStreet += additional;
            p.contributedTotal += additional;
            if (p.stack === 0) p.allIn = true;

            this.lastRaiseSize = Math.max(action.amount - this.currentBet, this.lastRaiseSize);
            this.currentBet = action.amount;

            this.needsToAct = new Set(this.playersWhoCanAct());
            this.needsToAct.delete(action.player);
        } else {
            throw new Error(`Unknown action type ${action.type}`);
        }

        this.advanceTurn();
    }

    getState(): TraceState {
        const pot = this.players.reduce((sum, p) => sum + p.contributedTotal, 0);
        const minRaiseTo =
            this.handOver || this.actingIdx === null ? 0 : this.currentBet + this.lastRaiseSize;

        return {
            street: STREETS[Math.min(this.streetIndex, 3)],
            pot,
            current_bet: this.currentBet,
            min_raise: minRaiseTo,
            board: this.board.map(toCardVector),
            dealer: 0,
            acting_idx: this.handOver ? null : this.actingIdx,
            hand_over: this.handOver,
            winner: this.winner,
            players: this.players.map((p, idx) => ({
                index: idx,
                stack: p.stack,
                folded: p.folded,
                contributed_street: p.contributedStreet,
                hole_cards: p.holeCards.map(toCardVector),
            })),
        };
    }

    getActionBounds(): ActionBounds {
        if (this.handOver || this.actingIdx === null) {
            return {
                action_type_space: 3,
                can_fold: false,
                can_call: false,
                can_raise: false,
                can_all_in: false,
                raise_amount_min: 0,
                raise_amount_max: 0,
            };
        }

        const p = this.players[this.actingIdx];
        const toCall = this.currentBet - p.contributedStreet;
        const minRaiseTo = this.currentBet + this.lastRaiseSize;
        const maxRaiseTo = p.stack + p.contributedStreet;
        const canRaise = p.stack > toCall && maxRaiseTo > this.currentBet;

        return {
            action_type_space: 3,
            can_fold: true,
            can_call: true,
            can_raise: canRaise,
            can_all_in: p.stack > 0,
            raise_amount_min: canRaise ? Math.min(minRaiseTo, maxRaiseTo) : 0,
            raise_amount_max: canRaise ? maxRaiseTo : 0,
        };
    }

    getTensorInput(playerIdx: number): number[] {
        const input: number[] = [];
        const pot = this.players.reduce((sum, p) => sum + p.contributedTotal, 0);
        const minRaiseTo =
            this.handOver || this.actingIdx === null ? 0 : this.currentBet + this.lastRaiseSize;

        input.push(pot / STARTING_STACK);
        input.push(this.currentBet / STARTING_STACK);
        input.push(minRaiseTo / STARTING_STACK);

        const streetOneHot = [0, 0, 0, 0];
        streetOneHot[Math.min(this.streetIndex, 3)] = 1;
        input.push(...streetOneHot);

        for (let i = 0; i < 5; i++) {
            if (i < this.board.length) {
                input.push(...toCardVector(this.board[i]));
            } else {
                input.push(0, 0);
            }
        }

        const self = this.players[playerIdx];
        input.push(self.stack / STARTING_STACK);
        input.push(self.folded ? 1 : 0);
        input.push(self.contributedStreet / STARTING_STACK);

        for (let i = 0; i < 2; i++) {
            if (i < self.holeCards.length) {
                input.push(...toCardVector(self.holeCards[i]));
            } else {
                input.push(0, 0);
            }
        }

        // absolute seat order, skipping self — matches env/poker.py's get_tensor_input exactly
        for (const p of this.players) {
            if (p === self) continue;
            input.push(p.stack / STARTING_STACK);
            input.push(p.folded ? 1 : 0);
            input.push(p.contributedStreet / STARTING_STACK);
        }

        return input;
    }
}
