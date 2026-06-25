import { Link } from "react-router-dom";
import styles from "../styles/css/navbar.module.css";
export function NavBar() {
  return (
    <div className={styles.sidebar}>
      <div>
        <h1>PokerRL</h1>

        <ul className={styles.link_list}>
          <li>
            <h2 className="montreal">
              <Link to="/available">Available Tables</Link>
            </h2>
          </li>
          <li>
            <h2 className="montreal">
              <Link to="/play">Play vs AI</Link>
            </h2>
          </li>
          <li>
            <h2 className="montreal">
              <Link to="/load">Load Table</Link>
            </h2>
          </li>
          <li>
            <h2 className="montreal">
              <Link to="/history">Table History</Link>
            </h2>
          </li>
        </ul>
      </div>

      <p>
        Made with ❤️ by{" "}
        <a
          href="https://github.com/ved-patel226/"
          target="_blank"
          rel="noopener noreferrer"
        >
          Ved Patel
        </a>
      </p>
    </div>
  );
}
