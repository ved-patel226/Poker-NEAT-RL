import styles from "../../styles/css/card.module.css";

export type CardProps = {
  rank?: string;
  suit?: string;
  back?: boolean;
};

export function Card({ rank, suit, back = false }: CardProps) {
  const suitLetter =
    suit === "clubs"
      ? "C"
      : suit === "diamonds"
        ? "D"
        : suit === "hearts"
          ? "H"
          : suit === "spades"
            ? "S"
            : "";

  if (back) {
    return (
      <img
        className={styles.card}
        src={import.meta.env.BASE_URL + "cards/2B.svg"}
        alt="Card back"
      />
    );
  }

  return (
    <img
      className={styles.card}
      src={import.meta.env.BASE_URL + "cards/" + rank + suitLetter + ".svg"}
      alt={rank + " of " + suit}
    />
  );
}
