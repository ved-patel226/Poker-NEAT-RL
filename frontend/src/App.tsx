import { useState, useEffect } from "react";
import "./App.css";

const RANK_MAP = [
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
const SUIT_MAP = ["♣", "♦", "♥", "♠"]; // Club, Diamond, Heart, Spade

function Card({ cardData }: { cardData: any }) {
  if (!cardData) return <div className="card unknown">?</div>;

  const rIdx = cardData.rank_one_hot.indexOf(1);
  const sIdx = cardData.suit_one_hot.indexOf(1);

  if (rIdx === -1 || sIdx === -1) return <div className="card unknown">?</div>;

  const rStr = RANK_MAP[rIdx];
  const sStr = SUIT_MAP[sIdx];
  const isRed = sIdx === 1 || sIdx === 2; // Diamond or Heart

  return (
    <div className={`card ${isRed ? "red" : "black"}`}>
      <span className="rank">{rStr}</span>
      <span className="suit">{sStr}</span>
    </div>
  );
}

function App() {
  const [gameState, setGameState] = useState<any>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");

    ws.onmessage = (event) => {
      setGameState(JSON.parse(event.data));
    };

    ws.onclose = () => console.log("WebSocket closed");

    return () => ws.close();
  }, []);

  if (!gameState) {
    return (
      <div className="loading">
        Waiting for game state from WebSocket (127.0.0.1:8000)...
      </div>
    );
  }

  return (
    <div className="poker-table">
      <div className="board-info">
        <h2>Street: {gameState.street}</h2>
        <h3>Pot: {gameState.pot}</h3>
        <div className="board-cards">
          {gameState.board.length > 0 ? (
            gameState.board.map((c: any, i: number) => (
              <Card key={i} cardData={c} />
            ))
          ) : (
            <span className="no-cards">No Community Cards</span>
          )}
        </div>
        {gameState.hand_over && (
          <div className="winner">
            Hand Over! Winner: Player {gameState.winner}
          </div>
        )}
      </div>

      <div className="players">
        {gameState.players.map((p: any) => (
          <div
            key={p.index}
            className={`player ${p.folded ? "folded" : ""} ${gameState.acting_idx === p.index ? "acting" : ""}`}
          >
            <div className="player-name">Player {p.index}</div>
            <div className="player-stack">Stack: {p.stack}</div>
            <div className="player-bet">Bet: {p.contributed_street}</div>
            <div className="hole-cards">
              {p.hole_cards.length > 0 ? (
                p.hole_cards.map((c: any, i: number) => (
                  <Card key={i} cardData={c} />
                ))
              ) : (
                <div className="hidden-cards">
                  <div className="card unknown"></div>
                  <div className="card unknown"></div>
                </div>
              )}
            </div>
            {p.folded && <div className="folded-badge">FOLDED</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
