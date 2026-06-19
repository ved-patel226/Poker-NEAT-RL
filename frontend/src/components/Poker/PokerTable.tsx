import type { ReactNode } from 'react';
import { Cards } from './Cards';
import type { CardProps } from './Card';
import type { CardVector, TracePlayer, TraceState } from './pokerTypes';


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