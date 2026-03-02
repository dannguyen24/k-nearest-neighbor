## Introduction

This is Assignment 1 of CS 484 – Data Mining. The goal is to implement a K-Nearest Neighbor (KNN) classifier to predict whether a movie review is positive (+1) or negative (-1).

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
- *Project Specifics:* Truncated SVD (like PCA for sparse matrices) is used to compress thousands of word features into a smaller set of components while preserving variance.

**Feature Subset Selection**
- *Primary Purpose:* A straightforward approach to dimensionality reduction that removes redundant or irrelevant features.
- *Applied to KNN Project?* Yes. KNN is highly sensitive to irrelevant features; removing them improves both accuracy and speed.
- *Project Specifics:* Stop-word removal is the primary selection method, filtering out semantically empty words such as "the" and "and."

**Feature Creation**
- *Primary Purpose:* Involves constructing new attributes that capture information more efficiently than the originals.
- *Applied to KNN Project?* Yes. KNN requires numerical input, so raw unstructured text must be converted into a numerical feature space.
- *Project Specifics:* Raw reviews are transformed into document vectors using TF-IDF weighting, assigning a numerical value to every unique term in the collection.

**Discretization & Binarization**
- *Primary Purpose:* Converts continuous attributes into ordinal categories or binary values (0 and 1).
- *Applied to KNN Project?* No. While KNN supports binary data via Jaccard similarity, sentiment analysis benefits more from the continuous weights provided by TF-IDF.

**Variable Transformation**
- *Primary Purpose:* Applies functions (e.g., log, Z-score normalization) to ensure features with large ranges do not disproportionately dominate the analysis.
- *Applied to KNN Project?* Yes. Without normalization, attributes with large numerical ranges would unfairly dominate distance calculations.
- *Project Specifics:* TF-IDF normalization is applied to document vectors so that longer reviews do not have an outsized influence on similarity measures compared to shorter ones.

---

## The Core Mental Model: What KNN Actually Does

Before diving into phases, it helps to understand KNN at a concrete level.

**KNN has no real "training" step.** Unlike neural networks, KNN does not learn weights or update parameters. It just stores everything and does all its work at prediction time. The training data IS the model.

Here is what the data looks like after preprocessing:

```
# After loading train_data.txt:
labels  = [+1, -1, +1, +1, -1, ...]   # 25,000 sentiment labels
reviews = ["great film ...", "boring waste ...", ...]  # 25,000 cleaned texts

# After TF-IDF:
X = a matrix of shape (25000, ~50000)
# Each row is one review.
# Each column is one unique word in the vocabulary.
# The value at row i, column j is the TF-IDF score of word j in review i.
# Most values are 0 (sparse matrix) since a review only uses a few thousand words.

# After SVD (dimensionality reduction):
X_reduced = a matrix of shape (25000, 100)
# Each row is still one review, but now compressed to 100 numbers instead of 50,000.
```

This matrix `X_reduced` paired with `labels` is your entire "model." To classify a new review:

1. Clean and vectorize the new review the same way (using the SAME fitted vectorizer and SVD)
2. Compute the distance/similarity between that one vector and all 25,000 rows in `X_reduced`
3. Find the k rows (training reviews) with the smallest distance (or highest similarity)
4. Look at their labels in the `labels` list — those are the neighbors' "experience"
5. Majority vote: if 4 out of 5 neighbors are +1, predict +1

**Distance vs Similarity:**
- Euclidean distance: lower = more similar (standard geometric distance)
- Cosine similarity: higher = more similar (measures the angle between vectors, ignores magnitude)
- Both are valid. You test both and pick whichever gives better cross-validation accuracy.

---

## Phase 1: Build the Vector Database

**What you are building:** A matrix where every row is one training review converted to numbers.

### Step 1 — Load the training data

```python
# labels = [+1, -1, +1, ...]  (25,000 integers)
# reviews = ["the film was great ...", ...]  (25,000 raw strings)
labels, reviews = load_data('train_data.txt')
```

### Step 2 — Clean each review

For every review string, apply:
- Lowercase everything
- Remove punctuation and numbers
- Remove stop words ("the", "a", "is" — words with no sentiment value)
- Stem each word ("running" → "run", "movies" → "movi")

Result: `clean_reviews` — same 25,000 strings, but stripped down.

### Step 3 — Convert text to numbers (vectorization)

`TfidfVectorizer` reads all 25,000 clean reviews and builds a vocabulary — every unique word becomes a column. Then it fills in TF-IDF scores.

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_reviews)
# X.shape → (25000, ~50000)
# This is your "vector database" — 25,000 rows, one per review
```

**Understanding X.shape → (25000, ~50000):** This describes the dimensions of the matrix — think of it like a table:

```
              word_0   word_1   word_2   ...   word_49999
review_0   [  0.0,     0.52,    0.0,    ...     0.0     ]
review_1   [  0.31,    0.0,     0.0,    ...     0.71    ]
review_2   [  0.0,     0.0,     0.88,   ...     0.0     ]
...
review_24999 [ 0.0,    0.14,    0.0,    ...     0.0     ]
```

- **25,000 rows** = one row per training review
- **~50,000 columns** = one column per unique word found across all reviews (the vocabulary)
- **Each cell value** = the TF-IDF score of that word in that review (0.0 if the word doesn't appear in that review)
- The `~` means "approximately" — the exact vocabulary size depends on how many unique words survive after stemming and stop-word removal

**Why mostly zeros?** Any single review only uses a few hundred words out of 50,000 possible ones, so most cells are 0. This is called a **sparse matrix** — Python stores it efficiently by only recording the non-zero values.

**The vectorizer must be fitted only on training data.** When you later process test reviews, you call `vectorizer.transform(test_reviews)` — not `fit_transform` — so test reviews are forced into the same vocabulary columns.

### Step 4 — Reduce dimensions with SVD

50,000 columns is too many. KNN distance calculations break down in very high dimensions (the "Curse of Dimensionality"). SVD compresses each vector down to a manageable size.

```python
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)
# X_reduced.shape → (25000, 100)
```

**Same rule as the vectorizer:** fit SVD only on training data, then apply the same transformation to test data with `svd.transform(X_test)`.

At this point you have:
- `X_reduced` — shape (25000, 100): the numeric representation of every training review
- `labels` — shape (25000,): the known answer (+1 or -1) for each row

---

## Phase 2: Set Up Cross-Validation (Parameter Search)

**The goal:** Find the best combination of parameters before ever touching `test_data.txt`.

**Why not just test on test_data.txt directly?** Because you have no ground truth labels for it. You can only submit to Gradescope 10 times per day, and each submission counts. Cross-validation lets you compare parameter settings using only the labeled training data.

---

### Analogy: Studying for an Exam

Imagine you have 500 pages of class notes and a real exam coming up (Gradescope).

You cannot use the real exam to practice — you only get one shot at it. So instead, you split your own notes into two parts:
- 400 pages to **study from**
- 100 pages to **quiz yourself** (pretend you haven't seen them)

You study the 400 pages, then quiz yourself on the 100. If you score 85%, great — that study method works. If you score 60%, something is wrong.

That is cross-validation. The 25,000 **labeled** training reviews are your notes. `test_data.txt` is the real exam. You never practice on the real exam — you hold it back until the very end.

---

### Analogy: The Grid Search (Testing All Combinations)

Now imagine you are not sure which study strategy works best:

- Strategy A: Read notes + highlight keywords
- Strategy B: Read notes + make flashcards
- Strategy C: Watch lecture videos + read notes

You want to find the best strategy, so you try **every one** and quiz yourself after each. You record your quiz score for each strategy. At the end, you pick the strategy with the highest score.

That is a **grid search**. In this project, your "strategies" are combinations of these four settings:

```
k_values         = [3, 5, 7, 11, 15]           # how many neighbors vote
pca_dims         = [50, 100, 200]               # how compressed the vectors are
vectorizer_types = ['binary', 'freq', 'tfidf']  # how words are scored
distance_metrics = ['cosine', 'euclidean']      # how similarity is measured
```

That gives 5 × 3 × 3 × 2 = **90 combinations** to try. For each one, you run cross-validation and record the average accuracy. At the end, pick the winner.

You are not guessing which settings work best — you are measuring all of them.

---

### What a "Fold" Is

A fold is just a chunk of your training data temporarily set aside to act as a mini test set.

With 25,000 reviews split into 5 folds:

```
Fold 1: reviews    0 – 4,999   (5,000 reviews)
Fold 2: reviews 5,000 – 9,999
Fold 3: reviews 10,000 – 14,999
Fold 4: reviews 15,000 – 19,999
Fold 5: reviews 20,000 – 24,999
```

Each fold takes a turn being the "quiz set." The other 4 folds are the "study set."

---

### What One Full Cross-Validation Run Looks Like (for one parameter combo)

Take the combination: `k=5, pca_dims=100, tfidf, cosine`

```
Round 1: Study on folds 2+3+4+5 → Quiz on fold 1 → accuracy = 84%
Round 2: Study on folds 1+3+4+5 → Quiz on fold 2 → accuracy = 87%
Round 3: Study on folds 1+2+4+5 → Quiz on fold 3 → accuracy = 83%
Round 4: Study on folds 1+2+3+5 → Quiz on fold 4 → accuracy = 86%
Round 5: Study on folds 1+2+3+4 → Quiz on fold 5 → accuracy = 85%

Average accuracy for this combo: (84+87+83+86+85) / 5 = 85%
```

Why average 5 rounds instead of just 1? Because one quiz might be unlucky — the 1,000 questions you picked happened to be on topics you studied less. Averaging 5 rounds gives a fairer, more reliable estimate.

---

### Why This Is Better Than a Single Split

If you just did one 80/20 split (20,000 train, 5,000 validate):
- Maybe that one validation set happened to have mostly easy reviews → score is inflated
- Maybe it had hard reviews → score is deflated
- You cannot tell which

With 5-fold cross-validation, every review spends exactly one round as a validation review. The average is much more reliable.

---

### The Key Rule: Never Leak Information

Every round, the vectorizer and SVD are **re-fitted from scratch on the training folds only**, then applied to the validation fold.

```
Round 1:
  - Fit vectorizer on folds 2+3+4+5  →  transform fold 1 using it
  - Fit SVD on folds 2+3+4+5         →  transform fold 1 using it
  - Run KNN, measure accuracy

Round 2:
  - Fit a fresh vectorizer on folds 1+3+4+5  →  transform fold 2
  - Fit a fresh SVD on folds 1+3+4+5         →  transform fold 2
  - Run KNN, measure accuracy
```

If you fitted the vectorizer on all 5 folds before splitting, the vocabulary would be built using words from the validation fold too — the model would have "seen" those reviews during setup, making accuracy look artificially higher than it really is. This is called **data leakage**.

---

## Phase 3: Implement KNN

For a single combination of parameters and a single fold rotation:

### Step 1 — Split into train and validation

```python
# train_X: matrix of shape (20000, vocab_size)  ← the 4 training folds
# train_y: labels for those 20000 reviews
# val_X:   matrix of shape (5000, vocab_size)   ← the 1 validation fold
# val_y:   true labels for those 5000 (used only to measure accuracy)
```

### Step 2 — Fit vectorizer + SVD on training folds only

```python
vectorizer = TfidfVectorizer()
train_X_tfidf = vectorizer.fit_transform(train_reviews)

svd = TruncatedSVD(n_components=n_dims)
train_X_reduced = svd.fit_transform(train_X_tfidf)

# Apply the same transform (do NOT refit) to validation:
val_X_tfidf   = vectorizer.transform(val_reviews)
val_X_reduced = svd.transform(val_X_tfidf)
```

### Step 3 — For each validation review, compute distance to all training reviews

```python
# For one validation vector v and all training vectors T:
# Euclidean: distances[i] = sqrt(sum((v - T[i])^2))
# Cosine similarity: sim[i] = dot(v, T[i]) / (|v| * |T[i]|)
```

You end up with an array of 20,000 distances (or similarities), one per training review.

### Step 4 — Find k nearest neighbors and vote

```python
# Sort by distance (ascending) or similarity (descending)
# Take the top-k indices
# Look up their labels in train_y
# Count +1 vs -1 votes
# Assign whichever has more votes as the prediction
```

### Step 5 — Measure accuracy

```python
# Compare predictions against val_y (the true labels you held out)
# accuracy = (number correct) / (total validation reviews)
```

Record this accuracy. Repeat for all K folds and average.

---

## Phase 4: Pick Best Parameters, Generate Output

### Step 1 — Select the winning combination

After the grid search completes, you have a table like:

| k  | pca_dims | vectorizer | distance   | avg_accuracy |
|----|----------|------------|------------|--------------|
| 5  | 100      | tfidf      | cosine     | 0.872        |
| 7  | 200      | tfidf      | cosine     | 0.869        |
| 3  | 50       | binary     | euclidean  | 0.821        |
| ...| ...      | ...        | ...        | ...          |

Pick the row with the highest avg_accuracy.

### Step 2 — Train the final model on all 25,000 training reviews

Re-fit the vectorizer and SVD on the entire training set (not just 4/5 of it). This is what "final training" means in KNN — it just means you are now storing all 25,000 vectors instead of 20,000.

```python
# Fit on ALL train data
final_vectorizer = TfidfVectorizer()
X_all = final_vectorizer.fit_transform(all_clean_reviews)

final_svd = TruncatedSVD(n_components=best_pca_dims)
X_all_reduced = final_svd.fit_transform(X_all)
```

### Step 3 — Process test_data.txt and generate predictions

```python
# Load and clean test reviews (no labels in this file)
test_reviews = load_test_data('test_data.txt')
clean_test = [preprocess(r) for r in test_reviews]

# Transform using the FINAL fitted vectorizer + svd (do not refit)
X_test_tfidf   = final_vectorizer.transform(clean_test)
X_test_reduced = final_svd.transform(X_test_tfidf)

# Run KNN for each test review against all 25,000 training vectors
predictions = knn_predict(X_test_reduced, X_all_reduced, labels, k=best_k)
```

### Step 4 — Write output.txt

```python
with open('output.txt', 'w') as f:
    for pred in predictions:
        f.write(str(pred) + '\n')
```

This file has 25,000 lines, each either +1 or -1. Submit to Gradescope.

### Step 5 — Report confusion matrix (for cross-validation results, not test)

Since test labels are hidden, use the confusion matrix on your cross-validation results to understand model behavior:

```
                  Predicted +1    Predicted -1
Actual +1     |   True Positive  | False Negative |
Actual -1     |   False Positive | True Negative  |
```

- **Precision** — of all predicted positive, how many were actually positive?
- **Recall** — of all actual positives, how many did we catch?
- **Accuracy** — (TP + TN) / total — the main metric for this assignment

---

## Summary: What Variable Controls What

| Variable | What it controls | Where it appears |
|---|---|---|
| `k` | How many neighbors vote on a prediction | KNN predict step |
| `n_components` (SVD) | How many dimensions to compress to | TruncatedSVD |
| Vectorizer type | How word importance is scored (binary/freq/tfidf) | TfidfVectorizer or CountVectorizer |
| Distance metric | How "closeness" is measured | KNN distance calculation |
| Stop word removal | Whether common words are filtered before vectorizing | preprocess() |
| K folds | How many splits for cross-validation | Cross-validation loop |

The cross-validation grid search tests all combinations of the first four rows to find which set produces the highest average accuracy on held-out training data.
