import { Link } from "react-router-dom";

export function Dashboard() {
    return (
        <div>
            <h1>Dashboard</h1>
            <p>Here, you can start games with your poker AI!</p>

            <div>
                <Link to="/play">Start Game</Link>
            </div>
        </div>
    )
}