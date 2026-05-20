# ExDBSCAN: Explaining DBSCAN with Counterfactual Reasoning

> **KDD '26** — Pernille Matthews, Lena Krieger, Tommaso Amico, Arthur Zimek, Thomas Seidl, and Ira Assent  
> Aarhus University · Forschungszentrum Jülich · LMU Munich · University of Southern Denmark

---

## Overview

**ExDBSCAN** is the first post-hoc counterfactual explanation method designed specifically for DBSCAN, the popular density-based clustering algorithm. While counterfactual explanations are well-established in supervised learning ("what is the smallest change to this input that would change the model's prediction?"), no prior work had brought rigorous, validity-guaranteed counterfactuals to density-based clustering — until now.

Given a data point and its DBSCAN assignment, ExDBSCAN answers:

> *"What minimal changes to this point would move it from noise into a cluster, or from one cluster into another?"*

ExDBSCAN generates multiple counterfactuals that are simultaneously **proximal** (close to the original point) and **diverse** (non-redundant alternatives), while respecting DBSCAN's density-connectivity structure. Every counterfactual is guaranteed to be valid by construction.

---

## Key Contributions

- **First method** to generate counterfactual explanations tailored for DBSCAN, covering both noise-to-cluster and cluster-to-cluster transitions.
- **Physics-inspired optimisation** that models the proximity–diversity trade-off as an energy minimisation problem, using:
  - *Coulomb-like electrostatic repulsion* to spread counterfactuals apart (diversity)
  - *Hooke's law spring attraction* to keep counterfactuals close to the explained point (proximity)
- **Density-aware graph representation** — an undirected weighted graph over core points whose shortest-path distances respect DBSCAN's notion of similarity, avoiding the pitfalls of naïve Euclidean distance.
- **Theoretical validity guarantee** (Theorem 3.1): every generated counterfactual is provably a member of the target cluster.
- **Support for non-actionable features**: counterfactuals can be constrained so that immutable attributes (e.g. age, genetic data) are never changed.
- Evaluated on **30 OpenML tabular datasets**, outperforming all baselines on proximity, diversity, and validity.

---

## Method

ExDBSCAN operates in two steps:

### 1. Reference Core Point Selection
For a target cluster, ExDBSCAN builds a weighted graph $G(V, E)$ whose vertices are the cluster's core points and whose edge weights are pairwise Euclidean distances. The shortest-path distance in $G$ captures density-connectivity. A set of $k$ reference core points is selected by greedily minimising the energy:

$$E_{C'} = \underbrace{\sum_{V_i, V_j \in C',\, j > i} \frac{1}{D(V_i, V_j)}}_{\text{repulsion (diversity)}} + \underbrace{\sum_{V_i \in C'} d(p, V_i)^2}_{\text{attraction (proximity)}}$$

This NP-hard optimisation is solved via a greedy approximation that empirically achieves within 5% of the optimum.

### 2. Counterfactual Construction
Each counterfactual $p'$ is placed at distance $\varepsilon$ from its reference core point $q$, in the direction of the original point $p$:

$$p' = p + (d_{pq} - \varepsilon)\,(q - p)/d_{pq}$$

By construction, $p'$ lies within the $\varepsilon$-neighbourhood of a core point in the target cluster, guaranteeing cluster membership (Theorem 3.1).

---

## Results

ExDBSCAN was evaluated against seven baselines across 30 datasets using three metrics:

| Metric | ExDBSCAN | Best Baseline |
|---|---|---|
| **Validity** | **100%** (guaranteed) | 100% (DiCE-Surrogate, ExDBSCAN Random) |
| **Proximity** | **Best** | DiCE-Surrogate (lower validity) |
| **Diversity** | **Best** (by large margin) | ExDBSCAN Random (worse proximity) |

Surrogate-based methods (DiCE-Surrogate, GS-Surrogate) and BayCon achieve less than 50% validity on average, because they optimise against surrogate decision boundaries that do not faithfully capture DBSCAN's discrete, density-connectivity-based assignments. ExDBSCAN is the only method that simultaneously achieves perfect validity, superior proximity, and top diversity.

---

## Baselines

| Method | Description |
|---|---|
| **BayCon** | Model-agnostic Bayesian counterfactual generator with Random Forest surrogate |
| **DiCE-Direct** | Diversity-aware optimisation querying DBSCAN directly |
| **DiCE-Surrogate** | DiCE optimising against a fitted surrogate classifier |
| **GS-Direct** | Growing Spheres on DBSCAN's discrete assignments |
| **GS-Surrogate** | Growing Spheres on a fitted surrogate classifier |
| **ExDBSCAN Random** | Ablation: random core-point selection with ExDBSCAN's construction step |

---

## Dataset Properties

Experiments were conducted on 30 tabular datasets from [OpenML](https://www.openml.org/). DBSCAN hyperparameters ($\varepsilon$, minPts) were selected via grid search maximising the DBCV clustering validity index.

| Dataset | Number of features | Number of samples | Ground truth clusters |
|---|---:|---:|---:|
| autoPrice | 15 | 159 | 2 |
| baskball | 4 | 96 | 2 |
| blood-transfusion | 4 | 748 | 2 |
| bodyfat | 14 | 252 | 2 |
| breast-w | 9 | 699 | 2 |
| chscase census2 | 7 | 400 | 2 |
| chscase census6 | 6 | 400 | 2 |
| chscase vine1 | 9 | 52 | 2 |
| confidence | 3 | 72 | 2 |
| diabetes | 8 | 768 | 2 |
| diabetes numeric | 2 | 43 | 2 |
| diggle table a1 | 4 | 48 | 2 |
| disclosure x noise | 3 | 662 | 2 |
| ecoli | 7 | 336 | 8 |
| glass | 9 | 214 | 7 |
| hayes-roth | 4 | 160 | 3 |
| heart-statlog | 13 | 270 | 2 |
| iris | 4 | 150 | 3 |
| liver-disorders | 6 | 345 | 2 |
| longley | 6 | 16 | 2 |
| machine cpu | 6 | 209 | 2 |
| mu284 | 10 | 284 | 2 |
| no2 | 7 | 500 | 2 |
| pm10 | 7 | 500 | 2 |
| prnn fglass | 9 | 214 | 6 |
| rabe 131 | 5 | 50 | 2 |
| sleep | 7 | 62 | 2 |
| strikes | 6 | 625 | 2 |
| vehicle | 18 | 846 | 4 |
| wine | 13 | 178 | 3 |

---

## Reproducibility

Dataset details and reproducible code are available at:  
**[http://anonymous.4open.science/r/ExDBSCAN-BDE3](http://anonymous.4open.science/r/ExDBSCAN-BDE3)**

---



---

## Acknowledgements

This work was partially funded by the Pioneer Centre for AI (DNRF grant P1), project W2/W3-108 of the Helmholtz Association Initiative and Networking Fund, and the Marie Skłodowska-Curie Doctoral Network RELAX-DN (EU Horizon Europe 2021–2027, grant agreement nr. 101072456).
