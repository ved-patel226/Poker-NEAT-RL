import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from .poker import Observation

app = FastAPI()

obs = Observation()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            game_state = obs.get_state()

            if "street" in game_state and hasattr(game_state["street"], "value"):
                game_state["street"] = game_state["street"].value

            await ws.send_json(game_state)

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("Client disconnected from stream")
