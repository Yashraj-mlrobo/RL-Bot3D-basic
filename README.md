# Adaptive Maze Agent (AMA): 3D Digital Twin 🏎️⚡

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Simulation](https://img.shields.io/badge/Engine-Ursina_3D-purple)
![AI](https://img.shields.io/badge/AI-Heuristic_Routing-brightgreen)

> *Teaching an AI to navigate a 3D maze is tough. Teaching it how to break out of an infinite mathematical "gravity trap" without breaking the simulation? That took some serious engineering.*

## 📌 Project Overview
The **Adaptive Maze Agent (AMA)** is a custom 3D digital twin built to visualize and stress-test Reinforcement Learning models in complex subterranean environments. 

Powered by the **Ursina Engine**, this repository consumes a pre-trained Proximal Policy Optimization (PPO) neural network and wraps it in a robust, self-correcting physics and UI/UX layer. It is designed to model the telemetry and decision-making required for autonomous subterranean drainage and pipe inspection.

---

## 🛠️ The Breakthrough: Hybrid Heuristic Overrides

A known limitation of pure Deep RL in sparse-reward mazes is the "Local Optima Trap" (the bot oscillates in corners because it is mathematically afraid of negative wall rewards). 

Instead of retraining the model, AMA implements a custom **Tabu Search & Chronological Backtracking** heuristic pipeline that actively supervises the neural network:

1. **Oscillation Detection:** The engine tracks the bot's position history. If 4 overlapping tiles are detected within 8 steps, the AI is temporarily locked out of the steering matrix.
2. **Physical Rewind:** The rover physically reverses 5 steps out of the dead end to return to a safe junction.
3. **Phantom Wall Injection:** The bot deploys temporary, glowing "Phantom Walls" onto the trap coordinates. This updates the mathematical matrix, forcing the AI's LiDAR to register the bad path as a solid physical barrier.
4. **Scan & Sprint:** The script scans for open corridors and forces a 3-step evasive sprint to break the gravity of the trap before safely handing control back to the AI.

---

## 🎮 The 3D Engine & UI

The simulation bridges the gap between raw tensor mathematics and an interactive visualizer:
* **Dynamic Chase Camera:** A third-person camera attached to the rover's chassis that pans and rotates perfectly with the steering matrix.
* **Real-Time Minimap:** A localized 2D dashboard HUD tracking procedural generation and target acquisition.
* **Breadcrumb Memory:** Fading visualizers that display the neural network's recent pathfinding history in real-time.

---

## 🚀 Installation & Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/Adaptive-Maze-Agent.git](https://github.com/yourusername/Adaptive-Maze-Agent.git)
cd Adaptive-Maze-Agent
```
2. Install dependencies:
```bash
pip install ursina numpy stable-baselines3 torch
```
3. Run the Simulation:
Ensure your pre-trained best_model.zip is located in the root directory, then execute the simulation:
```bash
python simulate3d.py
```
(Press 'R' during the simulation to instantly generate a new procedural maze).

Developed by Yash Raj Bhatnagar
