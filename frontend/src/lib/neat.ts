// NOTE:
// structure of this file is VERY similar to the python version...
// if you're here to learn from the codebase, visit the python version first, (its easier)
// /model/NEAT/*.py


export type GenomeJSON = {
    nodes: {
        ids: number[];
        types: number[]; // 0 input 1 hidden 2 output 3 bias
        activations: number[]; // 0 relu 1 tanh 2 sigmoid
    };
    connections: {
        indices: [number, number][];
        weights: number[];
        enabled: boolean[]; // most of the time everythings enabled... so idk if we acc need this or
    };
};


const ACTIVATIONS: ((x: number) => number)[] = [
    (x) => Math.max(0, x), // relu
    (x) => Math.tanh(x), // tanh
    (x) => 1 / (1 + Math.exp(-x)), // sigmoid
]

function computeTopologicalOrder(genome: GenomeJSON): number[] {
    const numNodes = genome.nodes.ids.length;
    const deps: Set<number>[] = Array.from({ length: numNodes }, () => new Set());

    genome.connections.indices.forEach(([inNode, outNode], i) => {
        if (genome.connections.enabled[i]) {
            deps[outNode].add(inNode);
        }
    });

    const order: number[] = [];
    const visited = new Set<number>();
    const visiting = new Set<number>();

    function visit(nodeIdx: number) {
        if (visited.has(nodeIdx)) return;
        if (visiting.has(nodeIdx)) return;
        visiting.add(nodeIdx);

        for (const dep of deps[nodeIdx]) {
            visit(dep);
        }

        visiting.delete(nodeIdx);
        visited.add(nodeIdx);
        order.push(nodeIdx);
    }

    for (let i = 0; i < numNodes; i++) {
        visit(i);
    }

    return order;
}

export function forward(genome: GenomeJSON, input: number[]): number[] {
    const numNodes = genome.nodes.ids.length;
    const values = new Array(numNodes).fill(0);

    genome.nodes.types.forEach((type, idx) => {
        if (type === 3) values[idx] = 1.0; // bias
    });

    let inputCursor = 0;
    genome.nodes.types.forEach((type, idx) => {
        if (type === 0) {
            values[idx] = input[inputCursor] ?? 0;
            inputCursor += 1;
        }
    });

    const incoming: { from: number, weight: number }[][] = Array.from(
        { length: numNodes },
        () => [],
    );

    genome.connections.indices.forEach(([inNode, outNode], i) => {
        if (genome.connections.enabled[i]) {
            incoming[outNode].push({ from: inNode, weight: genome.connections.weights[i] });
        }
    });

    const order = computeTopologicalOrder(genome);

    for (const nodeIdx of order) {
        const type = genome.nodes.types[nodeIdx];
        if (type === 0 || type === 3) continue; // skip input and bias nodes

        const conns = incoming[nodeIdx];
        if (conns.length === 0) continue;

        let total = 0;
        for (const conn of conns) total += values[conn.from] * conn.weight;

        const activationFn = ACTIVATIONS[genome.nodes.activations[nodeIdx]] ?? ACTIVATIONS[0];
        values[nodeIdx] = activationFn(total);
    }

    const outputs: number[] = [];
    genome.nodes.types.forEach((type, idx) => {
        if (type === 2) outputs.push(values[idx]);
    });

    return outputs;
}