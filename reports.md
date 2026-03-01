## Introduction

This is the assignment 1 of CS 484 - Data mining. This project aim to to implemennt K Nearest Neighbor Classification Algorithm. This is part of prediction task (classification) of Data minign. 
With corss validation: 
- Number of neighbors (k)
- Bag-of-words representation (binary, raw frequency, TF-IDF)
- Distance/similarity measures (at least two)
- Different feature sets (full vs. reduced using dimensionality reduction or feature selection)

### Knowledge Synthesis

We are living in the world of data. It's not difficult for one to think of how and where their data is being collected, such as through cookies on applications, satelites data, the Internet, social media, etc. And with data from which, give rise to the need of data mining, in order to derive knowledge, tendency, and behaviour. This process is called **Knowlwedge Discovery in Database (KDD)**

We can make into 4 main processes

- Data preprocessing: This process is essential, because we lack the computational resources to actually process through ALL data, which might includes outliers, repeated, redundant, or inconsistent data. Through this process, we can improve time, cost, and quality of the data before feeding it into the analysis step.

- Preprocessing Methods

| Preprocessing Method | Primary Purpose |
|---|---|
| **Aggregation** | Used for data reduction, changing the scale of data (e.g., city to state), and creating more "stable" data with less variability. |
| **Sampling** | The main technique for data selection, used because processing an entire dataset is often too expensive or time-consuming. |
| **Dimensionality Reduction** | Designed to reduce the number of features to combat the "Curse of Dimensionality," where data becomes sparse and distances become skewed. |
| **Feature Subset Selection** | A simple way to reduce dimensionality by removing redundant features (duplicates) or irrelevant features (those not helpful for the task). |
| **Feature Creation** | Involves building new attributes that capture information more efficiently than the originals, such as extracting edges from images or calculating density. |
| **Discretization & Binarization** | Used to convert continuous attributes into ordinal categories or binary values (0 and 1) to meet the requirements of specific classification algorithms. |
| **Variable Transformation** | Applies functions to all values of a variable, such as normalization or standardization, to ensure features with large ranges do not unfairly dominate the analysis. |

> In this assignmet, we are working with text data, therefore, I choose sampling, dimentionality reduction (wrapper approach, PCA) for preprocessing

> Wrapper Approaches: Use the target data mining algorithm as a black box to find the best subset of attributes. Select the feature subset that yields the best performance

- Data Mining: This core step applies algorithms to the prepared data. Its purpose is to build models for prediction (forecasting unknown or future values) or description (finding human-interpretable patterns like clusters).

> K-nearest neighbor: 

- Pattern/Model Evaluation: The purpose of this step is to assess the reliability of the model. It uses metrics like the confusion matrix and ROC curves to determine if the model generalizes well to unseen data.

- Knowledge Presentation: The final purpose is to communicate findings to the end-user, often through visualization
