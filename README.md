# 🚨 ASTraM-Nexus: Zero-Shot Prescriptive AI & Closed-Loop Traffic Orchestrator

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-green.svg)
![OSMnx](https://img.shields.io/badge/OSMnx-Dynamic%20Routing-orange.svg)

## 📌 The Problem
Bengaluru Traffic Police (BTP) manage 14,000 km of roads, but event-driven deployment remains experience-driven. When analyzing the provided BTP telemetry, we discovered a **98% sparsity rate** in critical operational fields and massive administrative lag masking actual clearance times. Standard predictive models fail under these real-world conditions.

## 💡 Our Solution
We abandoned standard regression and built a **sparse-data systems architecture**. We mathematically isolated acute traffic dispatch from civic infrastructure tickets and utilized p95-informed statistical capping to correct administrative time-travel errors. 

**ASTraM-Nexus** is a 4-pillar Command Center designed for the BTP Traffic Management Center (TMC):
1. **Rapid Incident Triage:** NLP semantic risk extraction + XGBoost Probabilistic Gridlock Classifier.
2. **Zero-Shot Dispatch & Wardrop Routing:** Prescriptive manpower allocation from nearest stations + dynamic OSRM real-road routing with Google Maps field-link generation.
3. **Planned Event Forecaster:** Predictive crowd-scale analyzer for permit-stage planning.
4. **Agentic Compliance & Feedback:** Simulated VLM (Vision-Language Model) auditing for physical barricade compliance + GEH-statistic post-event learning loop to monitor concept drift.

## ⚙️ Architecture

- **Data Pipeline:** `scripts/preprocess.py` (Handles missing data, caps outliers, engineers Target Variable).
- **ML Brain:** `scripts/train_model.py` (Trains the 3-Tier XGBoost Gridlock Probability Classifier).
- **Dispatch Engine:** `dispatcher.py` (Zero-shot heuristic matrix based on Haversine proximity).
- **Routing Engine:** `koramangala_diversion.py` (OSMnx/NetworkX shortest-path diversion calculation).
- **Frontend Command Center:** `app.py` (Streamlit + Folium Interactive Dashboard).

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/GridLockROUND-2.git
cd GridLockROUND-2
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the ASTraM-Nexus Command Center**
```bash
streamlit run app.py
```
*Note: The application features graceful degradation. It will utilize OSRM graphs and the pre-trained XGBoost model if present, but will seamlessly fall back to deterministic heuristics if dependencies fail.*
```
***

### Final Push
1. Run the `.gitignore` commands.
2. Reorganize the folders (and fix the 2-3 file paths in your code).
3. Add the `requirements.txt`.
4. Add the `README.md`.
5. Run: 
```bash
git add .
git commit -m "Refactor: Enterprise repo structure, added README and requirements"
git push
```
