# Multi-Agent Fishing
**COMP4900I / COMP5801G — Final Project**  
*SWEEEEEEEEET*

<img width="360" height="272" alt="cat_waving" src="https://github.com/user-attachments/assets/fa226d6d-997c-4e22-b1ab-da336e7ed1da" />

---

# 🌊 MARL Fishing Simulator Setup Guide

A multi-agent reinforcement learning simulation interface tracking fishing and patrol vessel interactions using Pygame, NumPy, and Xarray.

---

## 🚀 Quick Start Instructions

Follow these steps to configure your local Python environment and boot up the simulation viewer using **uv**.

### 1. Install `uv`
If you do not have `uv` installed on your machine yet, open your terminal (or PowerShell on Windows) and execute the appropriate platform command:

* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh | iex"
  ```
* **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh | sh
  ```

*Restart your terminal window after installation finishes to ensure the `uv` command is available in your PATH.*

### 2. Install the Virtual Environment
Navigate into your project repository directory. Running `uv sync` will automatically read your `pyproject.toml` and `uv.lock` files, create the `.venv` directory, and install the exact correct package versions:

```bash
# Navigate to the project root
cd Multi-Agent-Fishing

# Create the .venv and install all locked dependencies automatically
uv sync


## If it has issues finding the package you MAY need to run this (The period is necessary)
uv pip install -e .

```

### 3. VS Code Workspace Integration
Because **uv** names its environment `.venv`, VS Code will find it automatically. You just need to activate it:

1. Open your project folder in VS Code.
2. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on Mac) to open the Command Palette.
3. Type and choose **`Python: Select Interpreter`**.
4. Select the environment that points to your local **`.venv`** workspace directory.

### 4. Start the Simulation

**Through VS Code:**
Just run the `pygame_load_map.py` file like usual.

**Through UV in a terminal:**
```bash
uv run tools/pygame_load_map.py
```



---

## 🎮 Controls

Once the viewport initializes, you can control map parameters and toggle interactive telemetry displays using your keyboard and mouse:

* **Movement Matrix:** **Left-Click and Drag** or Hold **`W, A, S, D`** (or Arrow Keys) to pan the camera.
* **Camera Scaling:** Use your **Mouse Scroll Wheel** to zoom the camera.
* **Biomass Density Overlay:** Press **`H`** to toggle heat-mapped fish concentrations.
* **Global Text Labels:** Press **`L`** to quickly hide or show map labels.
* **Reset Position:** Press **`R`** to instantly snap the camera frame back to its original view.
* **Pause Simulation:** Press **`Spacebar`** to pause the simulation.
* **Measuring Ruler Tool:** **Right-Click and Drag** your cursor between any two points to show distance (km) and compass bearing degrees (°).


## Terrain Data

### Canadian Government data 
* **Source:** [Open Government Canada](https://canada.ca)
* **Details:** 10m resolution (7 GB file size).
* **Status:** Unused

### Gebco Data
* **Source:** GEBCO 2026 Dataset (15 arc-second intervals)
* **User-Defined Area Bounds:** North 55.0 to South 35.0 | West -135.0 to East -118.0
* **Attribution Requirement:**
  > GEBCO Bathymetric Compilation Group 2026 (2026). The GEBCO_2026 Grid - a continuous terrain model for oceans and land at 15 arc-second intervals. [doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa](https://doi.org)

---

## Ports & Harbor Mechanics

### Port Locations
* data generated via AI, seems reasonable but no guarantee of veracity.

### Vessel Spawn Logic
* Vessels are assigned a permanent **home harbor**.
* **Cycle:** Spawn at home harbor $\rightarrow$ Transit to hotspot $\rightarrow$ Fish $\rightarrow$ Return to home harbor.
* To prevent spawn-camping issues, consecutive new boats cycle through different starting harbors.

---

## Research & Simulation Parameters

### Agent Movement & Timings

#### Commercial Fishing Boats
* **Transit Speed:** 18–25 km/h
* **Fishing Speed:** 5–11 km/h
* **Timeline:** 3-hour transit to hotspot $\rightarrow$ 8-hour fishing window

#### Patrol Boats
* **Transit Speed:** 45–48 km/h
* **Deployable Dinghies:** 65 km/h max speed
* **Vision:** 20 km radar radius

---

## Fishing Hotspots

### Geographic Characteristics
* **Location:** Located 50–80 km offshore where the continental shelf drops off, creating nutrient-rich upwelling.
* **Target Species:** Salmon, Halibut, Tuna.
* **Dimensions:** Zones span roughly 5–20 km across and follow the natural curvature of the coastline.
* **Depth Profiles (Inferred from Global Fishing Watch data):**
  * Primary depth: 100–150 meters (a few kilometers inland from the continental shelf edge).
  * Deep-water exceptions: Around 500 meters, still on the continental shelf. (likely corresponding to specific gear or species).

### Notable Hotspots
* **La Perouse Bank:** 35–50 km offshore.
* **Cape Scott Deep Ocean:** 40–60 km offshore.
* **Tang.ɢwan — ḥačxʷiqak — Tsig̱is (Deep Offshore Marine Protected Area):** 80–150 km offshore.
