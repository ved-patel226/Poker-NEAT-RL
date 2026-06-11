import { BrowserRouter, Navigate, Routes, Route } from "react-router-dom";
import { Layout } from "./styles/Layout";
import { TexasHoldem } from "./components/Poker/TexasHoldem";
function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Navigate to="/load" replace />} />
            <Route path="/load" element={<TexasHoldem />} />
            <Route path="/available" element={<TexasHoldem />} />
            <Route path="*" element={<Navigate to="/load" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;
