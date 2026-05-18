import { Outlet } from "react-router-dom";
import { NavBar } from "../components/NavBar";

export function Layout() {
  return (
    <div className="layout">
      <NavBar />

      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
