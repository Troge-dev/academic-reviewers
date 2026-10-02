# DS312: Data Mining and Applications
## Module 6: Supervised vs. Unsupervised Methods
### Official Companion Study Guide

```text
========================================================================================
COURSE:        DS312 — Data Mining and Applications (BS Data Science 3rd Year)
MODULE:        Module 6: Supervised vs. Unsupervised Methods
INSTRUCTOR:    Nicole S. Menorias
SOURCE:        Module 6.ipynb (Complete & Grounded Study Guide)
========================================================================================
```

---

## 1. From Data Mining to Machine Learning

**Data mining** and **machine learning** overlap significantly and are often used interchangeably because both are fundamentally about extracting valuable insights from large datasets. However, they emphasize slightly different aspects:

* **Data Mining:** Centered on the **process of discovery** — finding patterns, associations, and anomalies that are already sitting in an existing dataset (e.g., clustering, association rules, and anomaly detection covered in Modules 1–5).
* **Machine Learning:** Centered on **building algorithms that learn** — models trained on historical data so they can make predictions or decisions on new, unseen data.

### Symbiosis in Practice
In real-world projects, the two disciplines borrow constantly from each other:
* A data mining project might use a machine learning algorithm to construct its predictive step.
* A machine learning project requires data mining's exploratory groundwork (data cleaning and exploratory data analysis) before any model can be trained.
* In industries such as **marketing** (customer segmentation), **finance** (fraud detection), and **healthcare** (diagnosis support), both disciplines work together.

---

## 2. The Bigger Picture: AI, Machine Learning, and Deep Learning

These three terms describe nested layers, each being a subset of the one before it:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ ARTIFICIAL INTELLIGENCE (Broadest Layer)                               │
│ Techniques that enable machines to mimic human behavior & decisions    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ MACHINE LEARNING (Subset of AI)                                │   │
│   │ Statistical methods so machines improve with experience        │   │
│   │   ┌────────────────────────────────────────────────────────┐   │   │
│   │   │ DEEP LEARNING (Subset of ML)                           │   │   │
│   │   │ Multi-layer neural networks for large, complex data    │   │   │
│   │   └────────────────────────────────────────────────────────┘   │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

| Layer | What It Is |
|---|---|
| **Artificial Intelligence (AI)** | The broadest layer — techniques that enable machines to mimic human behavior and decision-making in general. |
| **Machine Learning (ML)** | A subset of AI — uses statistical methods so machines **improve with experience**, rather than following only hand-written rules. |
| **Deep Learning (DL)** | A subset of ML — uses multi-layer neural networks, making it feasible to learn from very large, complex datasets (images, audio, raw text). |

### Concrete Illustration: Rule-Based Chess vs. Machine Learning
* A rule-based chess engine from the 1990s is **AI**, but it is **not Machine Learning** because it does not learn from data — it simply follows pre-programmed, hand-written rules.
* Picture three concentric circles: every deep learning system is a machine learning system, and every machine learning system is an AI system — but not the other way around.

---

## 3. Why Machine Learning Matters

* Provides organizations with insights into customer behavior and operational patterns (central to Google, Meta, Uber).
* Tackles problems difficult or impossible to solve with explicit hand-written rules (e.g., image recognition, speech recognition, natural language processing).
* **The Cat Recognition Analogy:** Instead of a programmer trying to enumerate every manual rule for "what a cat looks like," an ML model learns that pattern directly from thousands of labeled examples.
* **The Core Distinction:** Not every problem comes with labels — and that single difference splits machine learning into its major types.

---

## 4. The Four Types of Machine Learning

Machine learning is divided into four major types based on **the kind of information available to the model during learning**:

| Type | What the Model Receives | Main Idea |
|---|---|---|
| **Supervised Learning** | Labeled data | Learn from examples with known answers |
| **Unsupervised Learning** | Unlabeled data | Discover patterns or structure |
| **Semi-Supervised Learning** | Labeled and unlabeled data | Combine both approaches |
| **Reinforcement Learning** | Rewards and penalties | Learn through interaction and feedback |

---

### 4.1 Supervised Learning

* The model learns from **labeled data**.
* A labeled dataset contains:
  1. **Input features:** The information given to the model.
  2. **Target or output:** The known answer associated with each observation.
* **The Answer Key Analogy:** Supervised learning is like learning with an answer key. Because correct answers are available during training, we can compare predictions with actual answers to measure performance.

#### Email Spam Example
| Email | Label |
|---|---|
| *"Congratulations! You won ₱1,000,000!"* | **Spam** |
| *"Meeting at 2 PM tomorrow."* | **Not Spam** |
| *"Claim your free prize now!"* | **Spam** |

---

#### 4.1.1 Classification
Used when the target or output is a **category or class**.
* **Classification → predicting a category**
* **Examples:** Spam or Not Spam, Fraud or Not Fraud, Pass or Fail, Disease or No Disease, High / Medium / Low Risk.
* **Example Question:** *"Will this transaction be fraudulent?"* → Fraudulent / Not Fraudulent.
* **Common Classification Models:**
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. K-Nearest Neighbors (KNN)
  5. Support Vector Machine (SVM)
  6. Naive Bayes
  7. Neural Networks

---

#### 4.1.2 Regression
Used when the target or output is a **numerical value**.
* **Regression → predicting a numerical value**
* **Examples:** Predicting house prices, sales, temperature, income, demand.
* **Example Question:** *"How much will this house cost?"* → **₱4,500,000**.
* **Common Regression Models:**
  1. Linear Regression
  2. Polynomial Regression
  3. Decision Tree Regression
  4. Random Forest Regression
  5. Neural Network Regression

---

#### 4.1.3 Common Supervised Learning Models Summary
| Model | Basic Idea |
|---|---|
| **Linear Regression** | Models a numerical outcome using a linear relationship |
| **Logistic Regression** | Estimates the probability of belonging to a class |
| **Decision Tree** | Makes predictions through a sequence of decision rules |
| **Random Forest** | Combines predictions from multiple decision trees |
| **K-Nearest Neighbors (KNN)** | Uses nearby observations to make predictions |
| **Support Vector Machine (SVM)** | Finds a boundary that separates classes |
| **Naive Bayes** | Uses probability to classify observations |
| **Neural Network** | Learns complex patterns through interconnected layers |

---

### 4.2 Unsupervised Learning

Used when the data does **not have a known target or correct answer**. The model is not told what the correct output should be; instead, it analyzes the data to discover patterns, relationships, or structures.
* **Unsupervised learning → discovering structure in unlabeled data**

#### Customer Segmentation Example
A company has customer records with *Age*, *Income*, *Spending frequency*, and *Amount spent*, but **no column identifying the type of customer**. Unsupervised learning discovers natural customer groups:
* **Group 1:** Frequent, high-spending customers
* **Group 2:** Occasional, medium-spending customers
* **Group 3:** Infrequent, low-spending customers

---

#### 4.2.1 Clustering
Groups observations that are **similar to one another**.
* **K-Means:** Divides observations into a specified number of groups or clusters (for example, setting $K = 3$).
* **Hierarchical Clustering:** Creates a hierarchy of groups where observations are progressively combined into larger groups, producing a tree-like structure.
* **DBSCAN:** Identifies groups based on areas of high data density. Can also identify observations that do not belong to any dense group, which is useful for detecting noise or outliers.

---

#### 4.2.2 Association Rule Mining
Identifies items or events that frequently occur together.
* **Supermarket Example:** Customers who purchase **Bread → often also purchase Milk**.
* **Well-Known Algorithm:** **Apriori**.
* **Common Uses:** Market basket analysis, product recommendations, purchasing behavior analysis.

---

#### 4.2.3 Dimensionality Reduction
Represents data using fewer dimensions while attempting to preserve important information.
* **Example:** 500 variables $\longrightarrow$ 20 important components.
* **Well-Known Technique:** **Principal Component Analysis (PCA)**.
* **Benefits:** Simplifying complex datasets, visualization, reducing computational requirements, handling high-dimensional data.

---

#### 4.2.4 Common Unsupervised Learning Methods Summary
| Method | Main Purpose |
|---|---|
| **K-Means** | Groups observations into a specified number of clusters |
| **Hierarchical Clustering** | Builds a hierarchy of groups (tree-like structure) |
| **DBSCAN** | Finds dense groups and identifies noise/outliers |
| **Apriori** | Finds frequently associated items or events |
| **PCA** | Reduces the number of dimensions in a dataset |

---

### 4.3 Semi-Supervised Learning

Combines elements of supervised and unsupervised learning:
* Uses a **small amount of labeled data** and a **large amount of unlabeled data**.
* **Why Useful:** In many real-world situations, unlabeled data is easier, cheaper, and faster to collect than labeled data (e.g., medical images where only a few are labeled by specialist doctors).
* **Medical Image Example:** 10,000 medical images: 500 labeled by medical experts, 9,500 unlabeled.

#### 4.3.2 Common Semi-Supervised Approaches
1. **Self-Training:** The model learns from labeled data, uses its confident predictions to assign labels to unlabeled data, and adds them for further training.
2. **Pseudo-Labeling:** The model predicts labels for unlabeled data and treats highly confident predictions as temporary or pseudo-labels added to training data.
3. **Label Propagation:** Labels from known observations are propagated to similar unlabeled observations under the assumption that similar observations have similar labels.
4. **Semi-Supervised Neural Networks:** Trained using both labeled and unlabeled data to capture both ground-truth labels and underlying data distributions.

---

### 4.4 Reinforcement Learning

The agent learns by **interacting with an environment** through trial and error, rather than learning from a fixed dataset.
* **Interaction Loop:** State $\longrightarrow$ Action $\longrightarrow$ Reward $\longrightarrow$ Learning.
* Positive reward for desirable results; negative reward or penalty for undesirable results.

#### 4.4.1 Important Reinforcement Learning Concepts
| Term | Meaning |
|---|---|
| **Agent** | The learner or decision-maker |
| **Environment** | The world or system in which the agent operates |
| **State** | The current situation of the environment |
| **Action** | A decision made by the agent |
| **Reward** | Feedback received after an action |
| **Policy** | A strategy for selecting actions |

#### 4.4.2 Examples & Common Algorithms
* **Application Areas:** Game-playing systems, robotics, autonomous systems, resource allocation, control systems (sequential decision-making).
* **Common Algorithms:** **Q-Learning**, **SARSA**, **Deep Q-Networks (DQN)**, **Policy Gradient Methods**.

---

### 4.5 & 4.6 Comparing the Four Types

| Type | Learning Information | Main Goal | Example | Sample Models / Methods |
|---|---|---|---|---|
| **Supervised** | Labeled data | Predict a known output | Email spam prediction | Decision Tree, Random Forest, KNN, SVM, Linear Regression |
| **Unsupervised** | Unlabeled data | Discover patterns or structure | Customer segmentation | K-Means, DBSCAN, Hierarchical Clustering, PCA, Apriori |
| **Semi-Supervised** | Small labeled + large unlabeled dataset | Learn from both sources | Images where only some are labeled | Self-Training, Pseudo-Labeling, Label Propagation |
| **Reinforcement** | Rewards and penalties from interaction | Learn effective actions over time | Train agent to play a game | Q-Learning, SARSA, DQN, Policy Gradient |

#### Simple Conceptual Quotes
* **Supervised:** *"Here are examples with the correct answers. Learn to predict the answer for new examples."*
* **Unsupervised:** *"Here is the data. Find meaningful patterns or structure."*
* **Semi-Supervised:** *"Here are a few examples with answers and many examples without answers. Use both."*
* **Reinforcement:** *"Interact with the environment, take actions, receive feedback, and improve your decisions over time."*

---

## 5. The Accuracy–Interpretability Tradeoff

* **Accuracy:** *How well does the model make predictions?*
* **Interpretability:** *How easy is it for a human to understand why the model made that prediction?*
* **Graph Axes:**
  * **Vertical Axis (Y-axis):** Accuracy (higher = higher predictive accuracy).
  * **Horizontal Axis (X-axis):** Interpretability (farther right = easier to explain).
  * **Upper-Left:** High accuracy, but low interpretability (difficult to explain).
  * **Lower-Right:** High interpretability, but lower predictive accuracy.

### 5.2 How to Read the Models in the Figure
| Model | General Position | What It Means |
|---|---|---|
| **Neural Networks** | High accuracy, low interpretability | Learns very complex patterns, but difficult to explain why a prediction was made. |
| **Random Forest** | High accuracy, low–medium interpretability | Combines many trees; performs well, but understanding the entire model is difficult. |
| **Support Vector Machine** | Medium–high accuracy, low–medium interpretability | Finds a boundary separating groups; models complex relationships, but reasoning is not easy to explain. |
| **Graphical Models** | Medium accuracy, medium interpretability | Represents relationships using a graph, making some relationships easier to visualize. |
| **K-Nearest Neighbors (KNN)** | Medium accuracy, medium interpretability | Predicts based on nearby neighbors; reasoning becomes less clear with many variables. |
| **Decision Trees** | Medium accuracy, high interpretability | Predicts through simple decisions (*"Is age > 30?"* or *"Is income < ₱20,000?"*), making it easy to follow. |
| **Linear Regression** | Lower accuracy, high interpretability | Simple linear relationship ($Y = \beta_0 + \beta_1 X$); we can directly inspect $\beta_1$ to see input contribution. |
| **Classification Rules** | Lower accuracy, very high interpretability | Uses simple **IF–THEN rules**, making the decision process very easy to understand. |

### 5.3 The Student A vs. Student B Analogy
* **Student A:** Uses a complicated strategy considering many factors; accurate answer, but difficult to explain every step.
* **Student B:** Uses a simple rule (*"If X > 50, predict A; otherwise B"*); easy to understand, but may not capture all patterns.

---

## 6. Main Challenges in Machine Learning

Real-world ML projects face seven recurring obstacles:

1. **Poor quality of data:** Noisy, unclean data leads directly to inaccurate models (why Modules 4–5 emphasized cleaning & EDA).
2. **Underfitting:** The model is too simple to capture the real relationship between inputs and outputs.
   * *Fixes:* Adding more relevant features, increasing model complexity, or training longer.
3. **Overfitting:** The model memorized noise and bias in training data rather than the underlying pattern (performs great on training data, poorly on new data).
   * *Fixes:* Using more representative data, removing outliers, or choosing a simpler model.
4. **Overall process complexity:** ML is a young, fast-changing field full of trial and error; many interdependent steps can introduce error.
5. **Lack of training data:** Models often need very large amounts of data to learn reliable patterns.
6. **Slow implementation:** Training highly accurate models and monitoring/maintaining them takes substantial time and computing resources.
7. **Model decay as data grows:** A model that performs well today can quietly become less accurate as real-world data drifts, requiring regular monitoring and retraining.

---

## 7. Choosing an Approach: Practical Decision Guide

| If You Have... | ...And You Want To... | Consider |
|---|---|---|
| **Labeled historical outcomes** | Predict a future or unknown outcome | **Supervised learning** |
| **No labels at all** | Discover unknown groupings or patterns | **Unsupervised learning** |
| **A little labeled data, lots of unlabeled data** | Make the most of expensive-to-get labels | **Semi-supervised learning** |
| **No fixed dataset, but an environment to act in** | Learn a strategy through trial and error | **Reinforcement learning** |

---

## Quick Reference Summary Sheet

* **AI vs. ML vs. DL:** AI (mimic human behavior) $\supset$ ML (improve with experience from data) $\supset$ DL (multi-layer neural networks).
* **Supervised Learning:** Labeled data; Classification (category) vs. Regression (numerical value).
* **Unsupervised Learning:** Unlabeled data; Clustering (K-Means, Hierarchical, DBSCAN), Association Rules (Apriori), Dimensionality Reduction (PCA).
* **Semi-Supervised:** Small labeled + large unlabeled (Self-Training, Pseudo-Labeling, Label Propagation).
* **Reinforcement Learning:** Agent in Environment; State $\to$ Action $\to$ Reward $\to$ Learning; Q-Learning, SARSA, DQN.
* **Accuracy vs. Interpretability:** Neural Networks (upper-left, high accuracy/low interpretability) vs. Classification Rules & Linear Regression (lower-right, lower accuracy/high interpretability).
* **Underfitting vs. Overfitting:** Underfitting = too simple (fix: add features/complexity); Overfitting = memorized noise (fix: simpler model, remove outliers, representative data).
