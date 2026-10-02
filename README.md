# Multi-Agent Fishing
**COMP4900I / COMP5801G — Final Project**  
*SWEEEEEEEEET*

<img width="360" height="272" alt="cat_waving" src="https://github.com/user-attachments/assets/fa226d6d-997c-4e22-b1ab-da336e7ed1da" />

---

## Terrain Data

### Baseline Data (Unused)
* **Source:** [Open Government Canada](https://canada.ca)
* **Details:** 10m resolution (7 GB file size).
* **Status:** Discarded due to file size; requires significant downscaling.

### Updated Active Data
* **Source:** GEBCO 2026 Dataset (15 arc-second intervals)
* **User-Defined Area Bounds:** North 55.0 to South 35.0 | West -135.0 to East -118.0
* **Attribution Requirement:**
  > GEBCO Bathymetric Compilation Group 2026 (2026). The GEBCO_2026 Grid - a continuous terrain model for oceans and land at 15 arc-second intervals. [doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa](https://doi.org)

---

## Ports & Harbor Mechanics

### Port Locations
* Synthetic data generated via AI, likely modeled after official NOAA and DFO port locations.

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
  * Deep-water exceptions: Up to 500 meters (likely corresponding to specific gear/species like crab).

### Notable Hotspots
* **La Perouse Bank:** 35–50 km offshore.
* **Cape Scott Deep Ocean:** 40–60 km offshore.
* **Tang.ɢwan — ḥačxʷiqak — Tsig̱is (Deep Offshore Marine Protected Area):** 80–150 km offshore.
