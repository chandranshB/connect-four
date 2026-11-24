# connect-four

Connect Four played against an AlphaZero-style neural network, running completely client-side in WebAssembly via ONNX Runtime Web.

Live demo: https://chandranshB.github.io/connect-four/

## Overview

The engine uses a residual convolutional neural network with dual policy and value heads trained through Monte Carlo Tree Search self-play in PyTorch. The trained checkpoints are exported to ONNX and run entirely inside a browser Web Worker with WASM.

- Board representation: 64-bit bitboard state evaluation for fast bitwise move validation, win detection, and MCTS rollouts.
- Model tiers: 3 distinct checkpoints (Early, Mid, Pro) reflecting training progression and depth.
- Search: Client-side MCTS search with exploration constant tuning and live win-probability bars.
- Interface: Minimalist paper-style board with physical elevation, tactile drop animations, and SVG noise texture.

## Local Development

```bash
cd web
npm install
npm run dev
```

To run tests:

```bash
cd web
npm run test
```

## Training

The self-play training loop and ONNX export scripts are in `train/`:

```bash
cd train
pip install -r requirements.txt
python train.py
```
