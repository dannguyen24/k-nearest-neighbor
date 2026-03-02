## Introduction

This is Assignment 1 of CS 484 – Data Mining. The goal of this project is to implement the K-Nearest Neighbor (KNN) Classification Algorithm as part of a prediction (classification) task in data mining.

The following parameters are tuned using cross-validation:

- Number of neighbors (k)
- Bag-of-words representation (binary, raw frequency, TF-IDF)
- Distance/similarity measures (at least two)
- Different feature sets (full vs. reduced via dimensionality reduction or feature selection)

---

## Knowledge Synthesis

We live in a world saturated with data. It is easy to identify the many ways personal data is collected — through application cookies, satellite feeds, the Internet, social media, and more. This abundance of data gives rise to the need for **data mining**, a process used to derive knowledge, trends, and behavioral patterns. This broader process is known as **Knowledge Discovery in Databases (KDD)**.

KDD consists of four main stages:

- **Data Preprocessing:** An essential step, since processing all raw data is computationally infeasible. Raw data often contains outliers, duplicates, redundancies, and inconsistencies. Preprocessing improves the time, cost, and quality of data before it is fed into analysis.

- **Data Mining:** The core step, where algorithms are applied to the prepared data. Its purpose is to build models for either *prediction* (forecasting unknown or future values) or *description* (finding human-interpretable patterns such as clusters).

- **Pattern/Model Evaluation:** Assesses the reliability and generalizability of the model using metrics such as confusion matrices and ROC curves.

- **Knowledge Presentation:** Communicates findings to end users, typically through visualizations and summaries.

---

## Plan for This Assignment

### Preprocessing

**Aggregation**
- *Primary Purpose:* Used for data reduction, changing the scale of data (e.g., city → state), and producing more stable data with less variability.
- *Applied to KNN Project?* No. In sentiment analysis, each review is treated as a distinct object to classify; aggregating reviews would destroy individual-level sentiment signals.

**Sampling**
- *Primary Purpose:* The primary technique for data selection, used because processing an entire dataset is often too expensive or time-consuming.
- *Applied to KNN Project?* Yes. KNN has a high computational complexity of O(KN) per instance; reducing the number of training records (N) through sampling makes the algorithm significantly faster.
- *Project Specifics:* Stratified sampling is applied during K-Fold Cross-Validation to ensure each fold contains a representative mix of positive and negative reviews.

**Dimensionality Reduction**
- *Primary Purpose:* Reduces the number of features to combat the "Curse of Dimensionality," where data becomes sparse and distances lose their discriminative power.
- *Applied to KNN Project?* Yes. As the number of features grows, the distance to the nearest neighbor becomes indistinguishable from the average neighbor distance, degrading KNN's performance.
- *Project Specifics:* Principal Component Analysis (PCA) is used to compress thousands of word features into a smaller set of principal components while preserving variance.

**Feature Subset Selection**
- *Primary Purpose:* A straightforward approach to dimensionality reduction that removes redundant or irrelevant features.
- *Applied to KNN Project?* Yes. KNN is highly sensitive to irrelevant features; removing them improves both accuracy and speed.
- *Project Specifics:* Stop-word removal is the primary selection method, filtering out semantically empty words such as "the" and "and."

**Feature Creation**
- *Primary Purpose:* Involves constructing new attributes that capture information more efficiently than the originals (e.g., extracting edges from images, computing density).
- *Applied to KNN Project?* Yes. KNN requires numerical input, so raw unstructured text must be converted into a numerical feature space.
- *Project Specifics:* Raw reviews are transformed into document vectors using TF-IDF weighting, assigning a numerical value to every unique term in the collection.

**Discretization & Binarization**
- *Primary Purpose:* Converts continuous attributes into ordinal categories or binary values (0 and 1) to meet the requirements of specific algorithms.
- *Applied to KNN Project?* No. While KNN supports binary data via Jaccard similarity, sentiment analysis benefits more from the continuous weights provided by TF-IDF.

**Variable Transformation**
- *Primary Purpose:* Applies functions (e.g., log, Z-score normalization) to ensure features with large ranges do not disproportionately dominate the analysis.
- *Applied to KNN Project?* Yes. Without normalization, attributes with large numerical ranges would unfairly dominate distance calculations.
- *Project Specifics:* TF-IDF normalization is applied to document vectors so that longer reviews do not have an outsized influence on similarity measures compared to shorter ones.