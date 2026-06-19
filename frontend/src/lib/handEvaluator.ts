export type Card = { rank: number, suit: number }; // rank 2-14, suit 0-3

function combinations<T>(arr: T[], k: number): T[][] {
    const result: T[][] = [];
    const combination: T[] = [];

    function backtrack(start: number) {
        if (combination.length === k) {
            result.push([...combination]);
            return;
        }

        for (let i = start; i < arr.length; i++) {
            combination.push(arr[i]);
            backtrack(i + 1);
            combination.pop();
        }
    }

    backtrack(0);
    return result;
}

function evaluateFiveCardHand(cards: Card[]): number[] {
    const ranks = cards.map((c) => c.rank).sort((a, b) => b - a);
    const suits = cards.map((c) => c.suit);

    const isFlush = suits.every((s) => s === suits[0]);

    const counts = new Map<number, number>();
    ranks.forEach((r) => counts.set(r, (counts.get(r) ?? 0) + 1));

    const grouped = [...counts.entries()].sort((a, b) => b[1] - a[1] !== 0 ? b[1] - a[1] : b[0] - a[0]); // sort by count desc, then rank desc

    const uniqueRankDesc = [... new Set(ranks)].sort((a, b) => b - a);
    const ranksForStraight = uniqueRankDesc.includes(14) ? [...uniqueRankDesc, 1] : uniqueRankDesc;

    let straightHigh = -1;
    for (let i = 0; i <= ranksForStraight.length - 5; i++) {
        const slice = ranksForStraight.slice(i, i + 5);
        if (slice[0] - slice[4] === 4 && new Set(slice).size === 5) {
            straightHigh = slice[0];
            break;
        }
    }

    const isStraight = straightHigh !== -1;

    if (isStraight && isFlush) return [8, straightHigh]; // straight flush
    if (grouped[0][1] === 4) return [7, grouped[0][0], grouped[1][0]];
    if (grouped[0][1] === 3 && grouped[1][1] === 2) return [6, grouped[0][0], grouped[1][0]];
    if (isFlush) return [5, ...ranks];
    if (isStraight) return [4, straightHigh];
    if (grouped[0][1] === 3) return [3, grouped[0][0], ...grouped.slice(1).map(g => g[0])];
    if (grouped[0][1] === 2 && grouped[1][1] === 2) return [2, grouped[0][0], grouped[1][0], ...grouped.slice(2).map(g => g[0])];
    if (grouped[0][1] === 2) return [1, grouped[0][0], ...grouped.slice(1).map(g => g[0])];
    return [0, ...ranks];
}


export function compareScores(a: number[], b: number[]): number {
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
        const aval = a[i] ?? -1;
        const bval = b[i] ?? -1;
        if (aval !== bval) return aval - bval;
    }
    return 0;
}

export function bestHandScore(cards: Card[]): number[] {
    let best: number[] = [];
    for (const combo of combinations(cards, 5)) {
        const score = evaluateFiveCardHand(combo);
        if (!best || compareScores(score, best) > 0) best = score;
    }
    return best;
}

