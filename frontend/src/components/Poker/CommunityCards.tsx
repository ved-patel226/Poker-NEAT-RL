import { Card, type CardProps } from "./Card";
import styles from "../../styles/css/cards.module.css";

export type CommunityCardsProps = {
  cards?: CardProps[];
};

export function CommunityCards({ cards = [] }: CommunityCardsProps) {
  const communityCards = [...cards.slice(0, 5)];

  while (communityCards.length < 5) {
    communityCards.push({ back: true });
  }

  return (
    <div className={styles.cards}>
      {communityCards.map((cardProps, index) => (
        <Card
          key={`${cardProps.rank ?? "back"}-${cardProps.suit ?? "none"}-${index}`}
          {...cardProps}
        />
      ))}
    </div>
  );
}
