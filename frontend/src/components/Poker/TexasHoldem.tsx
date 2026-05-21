import { useEffect, useState, type ChangeEvent } from "react";
import type { CardProps } from "./Card";
import { Cards } from "./Cards";
import styles from "../../styles/css/texasholdem.module.css";

type CardVector = [number, number];

type TracePlayer = {
  index: number;
  stack: number;
  hole_cards: CardVector[];
  folded: boolean;
  contributed_street: number;
};

type TraceState = {
  players: TracePlayer[];
  board: CardVector[];
  pot: number;
  street: string;
  dealer: number;
  acting_idx: number | null;
  current_bet: number;
  min_raise: number;
  hand_over: boolean;
  winner: number | null;
  action_history?: unknown[];
};

type TraceAction = {
  player: number;
  street: string;
  type: number;
  amount: number;
};

type TraceEvent = {
  player: number;
  state: TraceState;
  action: TraceAction;
};

type TraceFile = {
  generation?: number;
  hand_count?: number;
  games?: TraceEvent[][];
};

type PlaybackEvent = TraceEvent & {
  handIndex: number;
  actionIndex: number;
};

const rankLabels = [
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "T",
  "J",
  "Q",
  "K",
  "A",
];

const suitLabels = ["clubs", "diamonds", "hearts", "spades"];

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function formatAmount(amount: number) {
  return `$${amount.toLocaleString("en-US")}`;
}

function decodeCard(vector: CardVector): CardProps {
  const [rankValue = 0, suitValue = 0] = vector;
  const rankIndex = clamp(Math.round(rankValue * 12), 0, rankLabels.length - 1);
  const suitIndex = clamp(Math.round(suitValue * 3), 0, suitLabels.length - 1);

  return {
    rank: rankLabels[rankIndex],
    suit: suitLabels[suitIndex],
  };
}

function decodeCards(cards: CardVector[] = []) {
  return cards.map(decodeCard);
}

function flattenTrace(trace: TraceFile) {
  return (trace.games ?? []).flatMap((hand, handIndex) =>
    (Array.isArray(hand) ? hand : []).map(
      (event, actionIndex) =>
        ({
          ...event,
          handIndex,
          actionIndex,
        }) as PlaybackEvent,
    ),
  );
}

// TODO: find a better way to do this...
// idrk im not feelin ts
function describeAction(action: TraceAction) {
  const actor = `Player ${action.player + 1}`;

  if (action.type === 0) {
    return `${actor} folded`;
  }

  if (action.type === 1) {
    return `${actor} checked or called`;
  }

  if (action.type === 2) {
    return `${actor} raised to ${formatAmount(action.amount)}`;
  }

  return `${actor} acted`;
}

function SeatCards({
  player,
  isActing,
}: {
  player: TracePlayer;
  isActing: boolean;
}) {
  return (
    <section
      className={`${styles.seat} ${isActing ? styles.seatActive : ""} ${
        player.folded ? styles.seatFolded : ""
      }`}
    >
      <Cards
        name={`Player ${player.index + 1}`}
        currentMoney={formatAmount(player.stack)}
        cards={decodeCards(player.hole_cards)}
      />
      <p className={styles.seatMeta}>
        {player.folded ? "Folded" : "In hand"} · Contributed{" "}
        {formatAmount(player.contributed_street)}
      </p>
    </section>
  );
}

export function TexasHoldem() {
  const [traceName, setTraceName] = useState<string>("");
  const [trace, setTrace] = useState<TraceFile | null>(null);
  const [events, setEvents] = useState<PlaybackEvent[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isPlaying || events.length === 0) {
      return;
    }

    if (currentStep >= events.length - 1) {
      setIsPlaying(false);
      return;
    }

    const currentEvent = events[currentStep];
    const delay = currentEvent.action.type === 2 ? 1400 : 900;

    const timeout = window.setTimeout(() => {
      setCurrentStep((value) => Math.min(value + 1, events.length - 1));
    }, delay);

    return () => window.clearTimeout(timeout);
  }, [currentStep, events, isPlaying]);

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const uploadedFile = event.target.files?.[0];

    if (!uploadedFile) {
      return;
    }

    setTraceName(uploadedFile.name);
    setError(null);

    try {
      const contents = await uploadedFile.text();
      const parsed = JSON.parse(contents) as TraceFile;
      const playbackEvents = flattenTrace(parsed);

      if (playbackEvents.length === 0) {
        throw new Error(
          "The uploaded file did not contain any playable hands.",
        );
      }

      setTrace(parsed);
      setEvents(playbackEvents);
      setCurrentStep(0);
      setIsPlaying(true);
    } catch (uploadError) {
      setTrace(null);
      setEvents([]);
      setCurrentStep(0);
      setIsPlaying(false);
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Unable to read that JSON file.",
      );
    }
  }

  const currentEvent = events[currentStep] ?? null;
  const currentState = currentEvent?.state ?? null;
  const players = currentState?.players ?? [];
  const splitIndex = Math.ceil(players.length / 2);
  const topRow = players.slice(0, splitIndex);
  const bottomRow = players.slice(splitIndex);
  const boardCards = decodeCards(currentState?.board ?? []);
  const currentActionText = currentEvent
    ? describeAction(currentEvent.action)
    : "Upload a JSON file to start the replay.";
  const currentHandLabel = currentEvent
    ? `Hand ${currentEvent.handIndex + 1}`
    : "Waiting for upload";
  const statusLabel = !trace
    ? "Waiting"
    : isPlaying
      ? "Playing"
      : currentStep >= events.length - 1
        ? "Finished"
        : "Paused";

  return (
    <div className={styles.poker}>
      <section className={styles.hero}>
        <div>
          <p className="montreal">Load playback</p>
          <h1>Upload a JSON trace</h1>
          <p>
            Select a poker-ai-game.json file and the replay will advance through
            each recorded state automatically. There is no betting input yet.
          </p>
        </div>

        <label className={styles.uploadButton}>
          <input
            className={styles.fileInput}
            type="file"
            accept="application/json,.json"
            onClick={(inputEvent) => {
              inputEvent.currentTarget.value = "";
            }}
            onChange={handleUpload}
          />

          <a>{traceName ? "Load a different file" : "Choose JSON file"}</a>
        </label>
      </section>

      {error ? <p className={styles.error}>{error}</p> : null}

      {trace && currentState ? (
        <section className={styles.tableShell}>
          <div className={styles.metaGrid}>
            <article className={styles.metaCard}>
              <p className={styles.metaLabel}>Generation</p>
              <h2 className={styles.metaValue}>{trace.generation ?? 0}</h2>
            </article>
            <article className={styles.metaCard}>
              <p className={styles.metaLabel}>Hand</p>
              <h2 className={styles.metaValue}>
                {currentStep + 1} / {events.length}
              </h2>
            </article>
            <article className={styles.metaCard}>
              <p className={styles.metaLabel}>Street</p>
              <h2 className={styles.metaValue}>{currentState.street}</h2>
            </article>
            <article className={styles.metaCard}>
              <p className={styles.metaLabel}>Status</p>
              <h2 className={styles.metaValue}>{statusLabel}</h2>
            </article>
          </div>

          <div className={styles.table}>
            <div className={styles.row}>
              {topRow.map((player) => (
                <SeatCards
                  key={player.index}
                  player={player}
                  isActing={currentState.acting_idx === player.index}
                />
              ))}
            </div>

            <div className={styles.centerStage}>
              <div className={styles.tableReadout}>
                <p className={styles.metaLabel}>{currentHandLabel}</p>
                <h1>{currentActionText}</h1>
                <p>
                  Pot {formatAmount(currentState.pot)} · Current bet{" "}
                  {formatAmount(currentState.current_bet)} · Min raise{" "}
                  {formatAmount(currentState.min_raise)}
                </p>
                {currentState.winner !== null ? (
                  <p>Winner: Player {currentState.winner + 1}</p>
                ) : (
                  <p>
                    Dealer: Player {currentState.dealer + 1} · Acting:{" "}
                    {currentState.acting_idx === null
                      ? "None"
                      : `Player ${currentState.acting_idx + 1}`}
                  </p>
                )}
              </div>

              <Cards cards={boardCards} />

              <p className={styles.traceNote}>
                {traceName} ·{" "}
                {currentState.hand_over ? "Hand complete" : "Hand in progress"}
              </p>
            </div>

            <div className={styles.row}>
              {bottomRow.map((player) => (
                <SeatCards
                  key={player.index}
                  player={player}
                  isActing={currentState.acting_idx === player.index}
                />
              ))}
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}
