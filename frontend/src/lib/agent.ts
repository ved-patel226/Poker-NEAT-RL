import type { ActionBounds } from "../components/Poker/pokerTypes";
import { forward, type GenomeJSON } from "./neat";

export type AgentAction = { type: number; amount: number };

export function chooseAction(genome: GenomeJSON, input: number[], bounds: ActionBounds): AgentAction {
    const out = forward(genome, input);
    const logits = out.slice(0, 3);

    const validMask = [bounds.can_fold, bounds.can_call, bounds.can_raise];
    const maskedLogits = logits.map((logit, i) => (validMask[i] ? logit : -Infinity));

    let typePicked = 0;
    let best = -Infinity;
    maskedLogits.forEach((logit, i) => {
        if (logit > best) {
            best = logit;
            typePicked = i;
        }
    });

    let amount = 0;
    if (typePicked === 2) {
        const sizeOut = 1 / (1 + Math.exp(-out[3]));
        const minR = bounds.raise_amount_min;
        const maxR = bounds.raise_amount_max;
        amount = Math.round(minR + sizeOut * (maxR - minR));
    }

    return { type: typePicked, amount };
}