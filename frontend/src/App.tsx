import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "./styles/Layout";
import { CommunityCards } from "./components/Poker/CommunityCards";

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route
              path="*"
              element={
                <CommunityCards
                  cards={[
                    { rank: "A", suit: "spades" },
                    { rank: "K", suit: "hearts" },
                    { rank: "Q", suit: "diamonds" },
                    { rank: "J", suit: "clubs" },
                    { rank: "T", suit: "spades" },
                  ]}
                />
              }
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;
