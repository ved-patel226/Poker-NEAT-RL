import type { ReactNode } from 'react';
import { Cards } from './Cards';
import type { CardProps } from './Card';
import type { CardVector, TracePlayer, TraceState } from './pokerTypes';
import styles from "../../styles/css/texasholdem.module.css"

export const rankLabels = [
    "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"
];

export const suitLabels = [
    "clubs", "diamonds", "hearts", "spades"
];

export function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
}

export function formatAmount(amount: number): string {
    return `$${amount.toLocaleString("en-US")}`;
}

export function decodeCard(vector: CardVector): CardProps {
    const [rankValue = 0, suitValue = 0] = vector;

    if (rankValue < 0 || suitValue < 0) {
        return { back: true };
    }

    const rankIndex = clamp(Math.round(rankValue * 12), 0, rankLabels.length - 1);
    const suitIndex = clamp(Math.round(suitValue * 3), 0, suitLabels.length - 1);

    return {
        rank: rankLabels[rankIndex],
        suit: suitLabels[suitIndex]
    };
}

export function decodeCards(cards: CardVector[] = []) {
    return cards.map(decodeCard);
}

export function SeatCards({ player, isActing }: { player: TracePlayer; isActing: boolean }) {
    return (
        <section
            className={`${styles.seat} ${isActing ? styles.seatActive : ""} ${player.folded ? styles.seatFolded : ""}`}
        >
            <Cards
                name={`Player ${player.index + 1}`}
                currentMoney={formatAmount(player.stack)}
                cards={decodeCards(player.hole_cards)}
            />
            <p className={styles.seatMeta}>
                {player.folded ? "Folded" : "In hand"} - Contributed{" "}{formatAmount(player.contributed_street)}
            </p>
        </section>
    );
}

export function PokerTableLayout({ state, centerContent }: { state: TraceState; centerContent: ReactNode }) {
    const players = state.players;
    const splitIndex = Math.ceil(players.length / 2);
    const topRow = players.slice(0, splitIndex);
    const bottomRow = players.slice(splitIndex);

    return (
        <div className={styles.table}>
            <div className={styles.row}>
                {topRow.map((player) => {
                    return <SeatCards key={player.index} player={player} isActing={state.acting_idx === player.index} />;
                })}
            </div>

            {centerContent}

            <div className={styles.row}>
                {bottomRow.map((player) => {
                    return <SeatCards key={player.index} player={player} isActing={state.acting_idx === player.index} />;
                })}
            </div>
        </div>
    )
}