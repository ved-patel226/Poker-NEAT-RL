import { useState, type ChangeEvent } from "react";
import { Cards } from "./Cards";
import styles from "../../styles/css/texasholdem.module.css";
import type { TraceAction, TraceEvent } from "./pokerTypes";
import { PokerTableLayout, decodeCards, formatAmount } from "./PokerTable";

type TraceFile = {
  generation?: number;
  hand_count?: number;
  games?: TraceEvent[][];
}

type PlaybackEvent = TraceEvent & {
  handIndex: number;
  actionIndex: number;
};

// TODO: put all these types into another file...


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


export function TexasHoldem() {
  const [traceName, setTraceName] = useState<string>("");
  const [trace, setTrace] = useState<TraceFile | null>(null);
  const [events, setEvents] = useState<PlaybackEvent[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Removed auto-cycling effect to require manual navigation via controls
  // useEffect(() => {
  //   if (!isPlaying || events.length === 0) {
  //     return;
  //   }

  //   if (currentStep >= events.length - 1) {
  //     setIsPlaying(false);
  //     return;
  //   }

  //   const currentEvent = events[currentStep];
  //   const delay = currentEvent.action.type === 2 ? 1400 : 900;

  //   const timeout = window.setTimeout(() => {
  //     setCurrentStep((value) => Math.min(value + 1, events.length - 1));
  //   }, delay);

  //   return () => window.clearTimeout(timeout);
  // }, [currentStep, events, isPlaying]);

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
      // setIsPlaying(false); // start paused so user controls playback
    } catch (uploadError) {
      setTrace(null);
      setEvents([]);
      setCurrentStep(0);
      // setIsPlaying(false);
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Unable to read that JSON file.",
      );
    }
  }

  const currentEvent = events[currentStep] ?? null;
  const currentState = currentEvent?.state ?? null;
  // const players = currentState?.players ?? [];
  // const splitIndex = Math.ceil(players.length / 2);
  // const topRow = players.slice(0, splitIndex);
  // const bottomRow = players.slice(splitIndex);
  // const boardCards = decodeCards(currentState?.board ?? []);
  const currentActionText = currentEvent
    ? describeAction(currentEvent.action)
    : "Upload a JSON file to start the replay.";
  const currentHandLabel = currentEvent
    ? `Hand ${currentEvent.handIndex + 1}`
    : "Waiting for upload";
  const statusLabel = !trace
    ? "Waiting"
    : currentStep >= events.length - 1
      ? "Finished"
      : "Paused";

  function goNext() {
    setCurrentStep((v) => Math.min(v + 1, events.length - 1));
  }

  function goPrevious() {
    setCurrentStep((v) => Math.max(v - 1, 0));
  }

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

          <PokerTableLayout
            state={currentState}
            centerContent={
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

                <Cards cards={decodeCards(currentState.board)} />

                <p className={styles.traceNote}>
                  {traceName} ·{" "}
                  {currentState.hand_over ? "Hand complete" : "Hand in progress"}
                </p>

                <div className={styles.playbackControls}>
                  <a
                    href="#"
                    className={styles.controlButton}
                    onClick={(e) => {
                      e.preventDefault();
                      goPrevious();
                    }}
                  >
                    &larr; Previous
                  </a>

                  <a
                    href="#"
                    className={styles.controlButton}
                    onClick={(e) => {
                      e.preventDefault();
                      goNext();
                    }}
                  >
                    Next &rarr;
                  </a>
                </div>
              </div>
            }
          />
        </section>
      ) : null
      }
    </div >
  );
}
