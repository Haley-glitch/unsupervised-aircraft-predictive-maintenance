# Unsupervised Predictive Maintenance for Aircraft Engines

> Detecting engine degradation without failure labels — unsupervised anomaly detection on the NASA C-MAPSS turbofan dataset using Autoencoders, PCA, and KMeans clustering.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 What Is This?

Aircraft engines fail gradually — sensors show subtle shifts in temperature, pressure, and fan speed long before anything breaks. Traditional predictive maintenance models need **labeled failure data** to learn from, but in the real world, such labels are rare, expensive, and inconsistent.

This project explores a fully **unsupervised approach**: instead of learning what failure looks like, we teach models what *healthy* looks like — and flag anything that deviates. Using the NASA C-MAPSS turbofan dataset, we build a pipeline that:

1. Learns a compressed representation of normal engine behavior via an **Autoencoder**
2. Derives a **Health Index (HI)** from reconstruction error — a continuous score that degrades as an engine approaches failure
3. Uses **PCA + KMeans** to cluster engine states into interpretable groups: *healthy*, *degrading*, and *near failure*

> 📄 This project was completed as part of **CMSC 471: Artificial Intelligence** at UMBC.

---

## 🗺️ Pipeline Overview

```
NASA C-MAPSS Dataset (FD001)
21 raw sensor readings per flight cycle
         │
         ▼
┌─────────────────────────┐
│  Feature Selection      │  ← Drop 9 flat/noisy sensors
│                         │    21 sensors → 12 informative
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Min-Max Normalization  │  ← Scale all features to [-1, 1]
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Autoencoder            │  ← Trained only on healthy data
│  (encoder → 5-dim       │    Reconstruction error = Health Index
│   bottleneck → decoder) │    Higher error = more degraded
└──────────┬──────────────┘
           │  Health Index per cycle
           ▼
┌─────────────────────────┐
│  PCA                    │  ← Reduce 12 features → 2D
│                         │    88% variance retained
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  KMeans Clustering      │  ← 3 clusters:
│                         │    Healthy / Degrading / Near Failure
└──────────┬──────────────┘
           │
           ▼
  Silhouette Score: 0.4902
  (moderately well-separated clusters)
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| NumPy / pandas | Data loading and manipulation |
| scikit-learn | Min-Max scaling, PCA, KMeans, Silhouette Score, Gradient Boosting, Random Forest |
| Keras / TensorFlow | Autoencoder neural network |
| Matplotlib / Seaborn | Visualization of health degradation, PCA clusters, HI distributions |
| Jupyter Notebook | Interactive analysis and presentation |

---

## 📂 Dataset

**NASA C-MAPSS FD001** (Commercial Modular Aero-Propulsion System Simulation)

- Simulated run-to-failure data from turbofan engines
- **Training set:** 100 engines, each run until failure
- **Test set:** 100 engines, stopped at some point before failure
- Each row = one flight cycle with:
  - 1 unit ID + 1 time cycle column
  - 3 operational settings
  - 21 sensor measurements

> ⬇️ Download from [Kaggle: NASA CMAPS Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place in `data/CMaps/`.

**Sensors removed (flat or uninformative):**
`sm_1, sm_5, sm_6, sm_10, sm_14, sm_16, sm_18, sm_19, oper_set3`
→ **21 sensors → 12 selected features**

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### Running the Notebook

```bash
git clone https://github.com/Haley-glitch/unsupervised-aircraft-predictive-maintenance.git
cd unsupervised-aircraft-predictive-maintenance

# Place NASA C-MAPSS data in:
# data/CMaps/train_FD001.txt
# data/CMaps/test_FD001.txt
# data/CMaps/RUL_FD001.txt

jupyter notebook aircraft_predictive_maintenance.ipynb
```

---

## ✨ Key Features & Methods

### 1. Feature Selection
Correlation analysis and visual inspection of sensor trends vs. RUL revealed 9 sensors with no useful signal (constant or random noise). These were dropped, leaving 12 informative sensors.

### 2. Autoencoder — Health Index
A neural network trained to compress and reconstruct healthy sensor data:

```
Input (12) → Dense(32) → Dense(16) → Bottleneck(5) → Dense(16) → Dense(32) → Output(12)
```

- Loss: Mean Squared Error
- Optimizer: Adam
- Trained for 20 epochs, batch size 128, 10% validation split

**Reconstruction error** serves as the Health Index (HI): the more an engine's sensor readings deviate from what healthy looks like, the higher the error — and the lower the HI score.

### 3. PCA + KMeans Clustering
- PCA reduces 12 features to 2 principal components, retaining **88% of variance**
- KMeans ($k=3$) groups engine cycles into: **Healthy**, **Degrading**, **Near Failure**
- Evaluated with Silhouette Score: **0.4902** — moderate cluster separation, consistent with real-world engine data where degradation is gradual and boundaries are not sharp

### 4. Supervised Baselines (for comparison)
The notebook also includes supervised RUL prediction models to benchmark against the unsupervised approach:

| Model | Train RMSE | Test RMSE |
|-------|-----------|-----------|
| Linear Regression | ~47 | ~52 |
| Gradient Boosting | ~18 | ~31 |
| Random Forest | ~22 | ~34 |

---

## 📊 Sample Output

**Health Index degradation over time** — each line is one engine unit. HI trends downward as the engine approaches end-of-life, with increasing volatility near failure.

**PCA cluster space** — three distinct regions emerge in 2D PCA space, corresponding to healthy (tight cluster), degrading (transitional), and near-failure (dispersed) engine states.

**Health Index distribution by cluster** — Cluster 0 concentrates near HI = 1.0 (healthy); Clusters 1 and 2 show progressively lower HI distributions.

*(See `aircraft_predictive_maintenance.ipynb` for all plots)*

---

## 🧠 Reflection

The most challenging aspect of this project was the **unsupervised framing itself** — without failure labels, there's no ground truth to optimize against directly. Choosing the right threshold for the Health Index, and deciding how many clusters $k$ to use for KMeans, required reasoning from domain knowledge and evaluating outputs visually rather than through a clean accuracy metric.

The Silhouette Score of 0.4902 initially felt disappointing, but it actually reflects something real: engine degradation is *gradual*, not a sharp jump between states. The moderate score is appropriate given the continuous nature of the underlying process. This was an important lesson in interpreting evaluation metrics in the context of the problem, not just as abstract numbers.

---

## 📚 References

- NASA PHM 2008 C-MAPSS Dataset — [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- Avellaneda-Stoikov (2008) — foundational HFT market making reference
- Li et al. — *Aeroengine health status evaluation based on PCA-KMeans and RBF neural network*, SPIE Proceedings
- Cartea, Jaimungal & Penalva (2015) — *Algorithmic and High-Frequency Trading*, Cambridge University Press
- CMSC 471 / CMSC 478, UMBC — course material on unsupervised learning

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
