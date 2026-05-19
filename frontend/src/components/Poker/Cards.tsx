import { Card, type CardProps } from "./Card";
import styles from "../../styles/css/cards.module.css";

export type CardsProps = {
  cards?: CardProps[];
  name?: string;
  currentMoney?: number | string;
  bigBlind?: boolean;
  smallBlind?: boolean;
};

export function Cards({
  cards = [],
  name,
  currentMoney,
  bigBlind,
  smallBlind,
}: CardsProps) {
  const a_Cards = [...cards.slice(0, 5)];
  const hasPlayerInfo =
    name !== undefined ||
    currentMoney !== undefined ||
    bigBlind !== undefined ||
    smallBlind !== undefined;

  return (
    <section className={styles.cardsContainer}>
      {hasPlayerInfo ? (
        <div className={styles.cardsText}>
          {name ? <p className={styles.cardsName}>{name}</p> : null}
          {currentMoney !== undefined ? (
            <p className={styles.cardsMeta}>Money: {currentMoney}</p>
          ) : null}
          {bigBlind && <p className={styles.cardsMeta}>Big Blind</p>}
          {smallBlind && <p className={styles.cardsMeta}>Small Blind</p>}
        </div>
      ) : null}
      <div className={styles.cards}>
        {a_Cards.map((cardProps, index) => (
          <Card
            key={`${cardProps.rank ?? "back"}-${cardProps.suit ?? "none"}-${index}`}
            {...cardProps}
          />
        ))}
      </div>
    </section>
  );
}
