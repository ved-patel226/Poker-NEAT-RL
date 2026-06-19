// Inference rn is to be run on the client side...
// i do NOT want to pay for a server
// also, the model is small enough that it should be able to run on the client side without too much issue (especially if we use something like ONNX or tfjs to optimize it for inference)
// .pkl won't work in the browser, so we need to convert it to a format that can be loaded and run in the browser 

import { BrowserRouter, Navigate, Routes, Route } from "react-router-dom";
import { Layout } from "./styles/Layout";
import { TexasHoldem } from "./components/Poker/TexasHoldem";
import { Dashboard } from "./components/Poker/Dashboard";

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Navigate to="/available" replace />} />
            <Route path="/load" element={<TexasHoldem />} />
            <Route path="/available" element={<Dashboard />} />
            <Route path="*" element={<Navigate to="/available" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;
