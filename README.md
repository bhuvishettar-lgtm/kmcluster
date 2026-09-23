# Iris Dataset K-Means Clustering

A Python-based machine learning pipeline demonstrating unsupervised clustering on the classic **Iris dataset** using the **K-Means algorithm**. The project covers data loading, train-test splitting, centroid initialization, model training, and cluster visualization.

---

## Algorithm Overview

K-Means identifies $k$ distinct clusters within an unlabeled dataset through an iterative optimization process:

1. **Initialization:** Select $k$ random data points as initial centroids ($\mu_1, \mu_2, \dots, \mu_k$).
2. **Assignment Step:** Assign each data point $x_i$ to its nearest centroid using Euclidean distance:
   $$\arg\min_{j} \Vert{} x_i - \mu_j \Vert{}^2$$
3. **Update Step:** Recalculate the centroid of each cluster based on the mean of all points assigned to it:
   $$\mu_j = \frac{1}{\vert{}S_j\vert{}} \sum_{x_i \in S_j} x_i$$
4. **Convergence:** Repeat Steps 2 and 3 until centroid coordinates stabilize (no shift in assignment occurs).

---

## Requirements & Dependencies

Ensure you have Python 3.8+ installed along with the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install dependencies via pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
