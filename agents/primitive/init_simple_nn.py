import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import asyncio
import uvicorn
from env.states import Action
from env.communication import app, obs
from model.simple_nn import SimpleNN


async def play_games():
    input_dim = 314
    output_dim = 4
    hidden_layers = [128, 64]

    agents = [SimpleNN(input_dim, output_dim, hidden_layers) for _ in range(6)]

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
                break

            state_tensor = obs.get_tensor_input(player_idx=actor_idx)

            with torch.no_grad():
                out = agents[actor_idx](state_tensor)

            action_logits = out[:3]
            action_type = torch.argmax(action_logits).item()

            bounds = obs.get_action_bounds()

            if action_type == 0 and not bounds["can_fold"]:
                action_type = 1
            if action_type == 1 and not bounds["can_call"]:
                action_type = 0 if bounds["can_fold"] else 2
            if action_type == 2 and not bounds["can_raise"]:
                action_type = 1 if bounds["can_call"] else 0

            if not bounds["can_fold"] and action_type == 0:
                action_type = 1
            if not bounds["can_call"] and action_type == 1:
                action_type = 0

            amount = 0
            if action_type == 2:
                min_r = bounds["raise_amount_min"]
                max_r = bounds["raise_amount_max"]
                ratio = torch.sigmoid(out[3]).item()
                amount = int(min_r + (max_r - min_r) * ratio)
                amount = max(int(min_r), min(int(max_r), amount))

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
