import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import random
import asyncio
import uvicorn
from env.states import Action
from env.communication import app, obs


async def play_games():
    num_hands = 10
    for hand_idx in range(num_hands):
        print(f"--- Hand {hand_idx + 1} ---")
        obs.reset()

        while True:
            state_dict = obs.get_state()
            if state_dict["hand_over"]:
                print(f"Hand over. Winner: {state_dict['winner']}")
                await asyncio.sleep(3)  # pause at the end of a hand to see the result
                break

            actor_idx = state_dict["acting_idx"]
            if actor_idx is None:
                print("DEBUG: actor_idx is None but hand_over is False!")
                break

            bounds = obs.get_action_bounds()

            # Determine valid actions
            valid_actions = []
            if bounds["can_fold"]:
                valid_actions.append(0)
            if bounds["can_call"]:
                valid_actions.append(1)
            if bounds["can_raise"]:
                valid_actions.append(2)

            # Fallback if no valid actions
            if not valid_actions:
                valid_actions = [0] if bounds["can_fold"] else [1]

            # Choose an action completely at random
            action_type = random.choice(valid_actions)

            amount = 0
            if action_type == 2:
                min_r = int(bounds["raise_amount_min"])
                max_r = int(bounds["raise_amount_max"])
                if max_r > min_r:
                    amount = random.randint(min_r, max_r)
                else:
                    amount = min_r

            action = Action(
                player=actor_idx,
                street=state_dict["street"],
                type=action_type,
                amount=amount,
            )

            action_names = {0: "FOLD", 1: "CALL/CHECK", 2: "RAISE"}

            try:
                obs.send_action(action)
                print(
                    f"Player {actor_idx} -> {action_names[action_type]} (amount: {amount})"
                )
            except Exception as e:
                print(
                    f"Player {actor_idx} -> Invalid Action: {e}. Falling back to FOLD/CHECK."
                )
                try:
                    if bounds["can_call"]:
                        obs.send_action(
                            Action(
                                player=actor_idx,
                                street=state_dict["street"],
                                type=1,
                                amount=0,
                            )
                        )
                    else:
                        obs.send_action(
                            Action(
                                player=actor_idx,
                                street=state_dict["street"],
                                type=0,
                                amount=0,
                            )
                        )
                except Exception as e2:
                    print(f"Fallback failure: {e2}")
                    break

            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(play_games())


def main():
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
