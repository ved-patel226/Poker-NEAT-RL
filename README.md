<p align="center">
	<img src="frontend/public/cards/AS.svg" />
</p>

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>PokerRL w/ NEAT</h1>
    </summary>
  </ul>
</div>


<p align="center">A reinforcement learning project to train agents via NEAT to play Texas hold 'em</p>

--- 

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2>Overview</h2>
    </summary>
  </ul>
</div>

PokerRL w/ NEAT trains Texas hold 'em agents by having genomes self-play at poker tables, scoring them across generations, and saving the best performers as checkpoints.

The Python trainer records TensorBoard metrics, checkpoints, and sampled hand traces. The frontend is a small Vite/React app for poker UI components and visualization.

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2>Repository Layout</h2>
    </summary>
  </ul>
</div>

- `env/` - poker state, action, and observation code.
- `model/` - NEAT implementation, checkpointing, and logging.
- `scripts/train_poker.py` - main training entry point.
- `frontend/` - React + TypeScript UI.
- `runs/` - training outputs, TensorBoard logs, and checkpoints.

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2>Requirements</h2>
    </summary>
  </ul>
</div>

- Python 3.10+.
- Node.js 20+.
- `requirements.txt` soon

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2>Setup from source</h2>
    </summary>
  </ul>
</div>

From the repository root:

```bash
pip install -r requirements.txt
```

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2>Train</h2>
    </summary>
  </ul>
</div>

Run the trainer from the repository root so it can load `config.yaml`:

```bash
python scripts/train_poker.py
```

The trainer reads `config.yaml` and maps it into enviorment variables before training starts.

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2>Frontend</h2>
    </summary>
  </ul>
</div>

```bash
cd frontend
npm run dev
```



<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2>Outputs</h2>
    </summary>
  </ul>
</div>

Each training run writes to a timestamped folder under `runs/`.

- TensorBoard logs: `runs/neat_<timestamp>/`
- Checkpoints: `runs/neat_<timestamp>/checkpoints/`
- Sampled hand traces: `poker-ai-game.json`

The JSON file is overwritten on each logging pass with the current hand samples.