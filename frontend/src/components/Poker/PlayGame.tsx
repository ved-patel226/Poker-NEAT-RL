import { useEffect, useRef, useState } from "react";
import { PokerTableLayout, decodeCards, formatAmount } from "./PokerTable";
import { Cards } from "./Cards";
import type { ActionBounds, TraceState } from "./pokerTypes";
import { PokerEngine } from "../../lib/pokerEngine";
import { chooseAction } from "../../lib/agent";
import type { GenomeJSON } from "../../lib/neat";
import styles from "../../styles/css/texasholdem.module.css";

type ManifestEntry = { label: string; path: string };

function hideOpponentCards(state: TraceState, humanSeat: number): TraceState {
    if (state.hand_over) return state;
    return {
        ...state,
        players: state.players.map((p) =>
            p.index === humanSeat
                ? p
                : { ...p, hole_cards: [[-1, -1], [-1, -1]] as [number, number][] },
        ),
    };
}

function runAiAutoplay(engine: PokerEngine, genomes: GenomeJSON[], humanSeat: number): void {
    let state = engine.getState();

    while (!state.hand_over && state.acting_idx !== null && state.acting_idx !== humanSeat) {
        const seat = state.acting_idx;
        const bounds = engine.getActionBounds();
        const input = engine.getTensorInput(seat);

        try {
            const action = chooseAction(genomes[seat], input, bounds);
            engine.sendAction({ player: seat, type: action.type, amount: action.amount });
        } catch {
            try {
                if (bounds.can_call) {
                    engine.sendAction({ player: seat, type: 1, amount: 0 });
                } else {
                    engine.sendAction({ player: seat, type: 0, amount: 0 });
                }
            } catch {
                break;
            }
        }

        state = engine.getState();
    }
}

export function PlayGame() {
    const [manifest, setManifest] = useState<ManifestEntry[]>([]);
    const [selectedPath, setSelectedPath] = useState<string>("");
    const [humanSeat, setHumanSeat] = useState(0);
    const [genomes, setGenomes] = useState<GenomeJSON[] | null>(null);
    const [state, setState] = useState<TraceState | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [raiseAmount, setRaiseAmount] = useState(0);
    const engineRef = useRef<PokerEngine | null>(null);

    useEffect(() => {
        fetch(import.meta.env.BASE_URL + "genomes/manifest.json")
            .then((res) => res.json())
            .then((data: ManifestEntry[]) => {
                setManifest(data);
                if (data.length > 0) setSelectedPath(data[0].path);
            })
            .catch(() => setError("Failed to load genome manifest"));
    }, []);

    const bounds: ActionBounds | null =
        state && !state.hand_over && state.acting_idx === humanSeat
            ? (engineRef.current?.getActionBounds() ?? null)
            : null;

    useEffect(() => {
        if (bounds) setRaiseAmount(bounds.raise_amount_min);
    }, [bounds]);

    async function startGame() {
        setError(null);
        try {
            const res = await fetch(import.meta.env.BASE_URL + `genomes/${selectedPath}`);
            const genomeJson: GenomeJSON = await res.json();
            const loadedGenomes = Array.from({ length: 6 }, () => genomeJson);
            setGenomes(loadedGenomes);

            const engine = new PokerEngine();
            engineRef.current = engine;
            runAiAutoplay(engine, loadedGenomes, humanSeat);
            setState(engine.getState());
        } catch {
            setError("Failed to load genome or start game");
        }
    }

    function handleAction(type: number, amount: number) {
        const engine = engineRef.current;
        if (!engine || !genomes) return;
        try {
            engine.sendAction({ player: humanSeat, type, amount });
            runAiAutoplay(engine, genomes, humanSeat);
            setState(engine.getState());
        } catch (e) {
            setError(e instanceof Error ? e.message : "Action failed");
        }
    }

    function handleNextHand() {
        const engine = engineRef.current;
        if (!engine || !genomes) return;
        engine.reset();
        runAiAutoplay(engine, genomes, humanSeat);
        setState(engine.getState());
    }

    if (!state) {
        return (
            <div className={styles.poker}>
                <section className={styles.hero}>
                    <h1>Play vs AI</h1>
                    {error ? <p className={styles.error}>{error}</p> : null}
                    <label>
                        Genome:{" "}
                        <select value={selectedPath} onChange={(e) => setSelectedPath(e.target.value)}>
                            {manifest.map((m) => (
                                <option key={m.path} value={m.path}>
                                    {m.label}
                                </option>
                            ))}
                        </select>
                    </label>
                    <label>
                        Your seat:{" "}
                        <select value={humanSeat} onChange={(e) => setHumanSeat(Number(e.target.value))}>
                            {[0, 1, 2, 3, 4, 5].map((seat) => (
                                <option key={seat} value={seat}>
                                    Seat {seat}
                                </option>
                            ))}
                        </select>
                    </label>
                    <button onClick={startGame} disabled={!selectedPath}>
                        Start Game
                    </button>
                </section>
            </div>
        );
    }

    const displayState = hideOpponentCards(state, humanSeat);
    const humanPlayer = state.players[humanSeat];
    const canCheck = bounds && state.current_bet === humanPlayer.contributed_street;

    const centerContent = (
        <div className={styles.centerStage}>
            <div className={styles.tableReadout}>
                <p>
                    Pot {formatAmount(state.pot)} · Current bet {formatAmount(state.current_bet)} · Min
                    raise {formatAmount(state.min_raise)} · Your seat {humanSeat}
                </p>
            </div>

            <Cards cards={decodeCards(state.board)} />

            {state.hand_over ? (
                <>
                    <p>Winner: Seat {state.winner}</p>
                    <button onClick={handleNextHand}>Next Hand</button>
                </>
            ) : bounds ? (
                <div>
                    <button onClick={() => handleAction(0, 0)} disabled={!bounds.can_fold}>
                        Fold
                    </button>
                    <button onClick={() => handleAction(1, 0)} disabled={!bounds.can_call}>
                        {canCheck ? "Check" : "Call"}
                    </button>
                    {bounds.can_raise ? (
                        <>
                            <input
                                type="range"
                                min={bounds.raise_amount_min}
                                max={bounds.raise_amount_max}
                                value={raiseAmount}
                                onChange={(e) => setRaiseAmount(Number(e.target.value))}
                            />
                            <button onClick={() => handleAction(2, raiseAmount)}>
                                Raise to {formatAmount(raiseAmount)}
                            </button>
                        </>
                    ) : null}
                </div>
            ) : (
                <p>AI is thinking...</p>
            )}
        </div>
    );

    return (
        <div className={styles.poker}>
            {error ? <p className={styles.error}>{error}</p> : null}
            <PokerTableLayout state={displayState} centerContent={centerContent} />
        </div>
    );
}
