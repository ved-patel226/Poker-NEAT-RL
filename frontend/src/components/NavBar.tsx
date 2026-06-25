import styles from "../styles/css/navbar.module.css";
export function NavBar() {
  return (
    <div className={styles.sidebar}>
      <div>
        <h1>PokerRL</h1>

        <ul className={styles.link_list}>
          <li>
            <h2 className="montreal">
              <a href="/available">Available Tables</a>
            </h2>
          </li>
          <li>
            <h2 className="montreal">
              <a href="/play">Play vs AI</a>
            </h2>
          </li>
          <li>
            <h2 className="montreal">
              <a href="/load">Load Table</a>
            </h2>
          </li>
          <li>
            <h2 className="montreal">
              <a href="/history">Table History</a>
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
