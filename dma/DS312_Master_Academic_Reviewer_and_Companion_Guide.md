# DS312: Data Mining & Applications
## Master Academic Reviewer, Architectural Companion, and Exam Defense Guide
### Comprehensive Curriculum Coverage: Modules 1 to 5 (Foundations, Analytics, Techniques, EDA, & Text Mining)

```
====================================================================================================
  ██████╗ ███████╗██████╗  ██╗██████╗     ██████╗  █████╗ ████████╗ █████╗     ███╗   ███╗██╗███╗   ██╗
  ██╔══██╗██╔════╝╚════██╗███║╚════██╗    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ████╗ ████║██║████╗  ██║
  ██║  ██║███████╗ █████╔╝╚██║ █████╔╝    ██║  ██║███████║   ██║   ███████║    ██╔████╔██║██║██╔██╗ ██║
  ██║  ██║╚════██║ ╚═══██╗ ██║██╔═══╝     ██║  ██║██╔══██║   ██║   ██╔══██║    ██║╚██╔╝██║██║██║╚██╗██║
  ██████╔╝███████║██████╔╝ ██║███████╗    ██████╔╝██║  ██║   ██║   ██║  ██║    ██║ ╚═╝ ██║██║██║ ╚████║
  ╚═════╝ ╚══════╝╚═════╝  ╚═╝╚══════╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝
  ACADEMIC COMPANION GUIDE | FOUR-TIER PEDAGOGICAL DEEP DIVE | FULL CURRICULUM SYNTHESIS (MODULES 1-5)
====================================================================================================
```

---

## Executive Preface & Curriculum Architecture

This master reviewer serves as the definitive theoretical and practical companion for **DS312: Data Mining & Applications**, synthesizing the foundational concepts, mathematical formulations, algorithmic workflows, and diagnostic decision frameworks across **Module 1 (Introduction & Evolution)**, **Module 2 (Types of Data Analytics)**, **Module 3 (Data Mining Techniques)**, **Module 4 (Structured Data Preprocessing & EDA)**, and **Module 5 (Unstructured Data Preprocessing & Clinical NLP)**.

### The Four-Tier Analytical Architecture
Every unit within this companion is structured across four rigorous tiers:
1. **Academic Context & Theoretical Deep Dive:** The mathematical principles, probability models, and underlying data structures (e.g., IQR fences, Little's MCAR chi-square, Dirichlet priors in LDA, Cosine geometry).
2. **Key Definition Glossary:** Precise, authoritative definitions of specialized data science terminology.
3. **Applied Industrial Scenarios & Real-World Failure Modes:** Production data pipeline failures, target leakage, imputation traps, and clinical risk miscalculations.
4. **Diagnostic Reviewer Questions & Defense Audits:** High-yield identification questions, scenario decision audits, and mathematical formula cheat sheets.

---

# MODULE 1: Introduction to Data Mining & Historical Evolution

## 1. Academic Context & Theoretical Deep Dive

### 1.1 The Definition of Data Mining
Data Mining is formally defined as the **non-trivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns from massive, heterogeneous data repositories** (Fayyad et al., 1996 - The KDD Process).
* **Non-trivial:** Requires inference, computation, and algorithmic search rather than simple SQL lookups or arithmetic calculations.
* **Valid:** Patterns must hold true on new, unseen data with statistical significance (generalizability).
* **Novel:** Patterns must uncover non-obvious relationships that humans could not easily deduce via intuition.
* **Potentially Useful:** Patterns must lead to actionable operational or strategic interventions (business or clinical utility).
* **Understandable:** Patterns must be interpretable by human domain experts.

### 1.2 Historical Evolution Timeline
Modern Data Mining sits at the convergence of Database Systems, Statistics, Machine Learning, and High-Performance Computing:
1. **1960s (Data Collection):** Flat file storage, magnetic tapes, primitive file access systems.
2. **1970s–1980s (Relational Databases & SQL):** Edgar F. Codd invents the Relational Model (RDBMS). Introduction of SQL; queries answer deterministic factual questions (*"Select all patients with Type 2 Diabetes"*).
3. **1990s (Data Warehousing & OLAP):** Inmon & Kimball architectures, multidimensional data cubes, Online Analytical Processing (OLAP). Emergence of Knowledge Discovery in Databases (KDD).
4. **2000s (Big Data Explosion):** Web 2.0, Hadoop MapReduce, distributed file systems (HDFS). The Three V's (Volume, Velocity, Variety).
5. **2010s–Present (Cloud & Advanced Analytics):** GPUs, Apache Spark, Deep Neural Networks, AutoML, Real-Time Streaming, MLOps.

### 1.3 The Three Modern Drivers
Data Mining became ubiquitous due to three simultaneous technological tailwinds:
* **Data Explosion:** Proliferation of IoT sensors, electronic health records (EHR), clickstreams, mobile devices, and transactional databases.
* **Hardware Cost Plummet:** The dramatic decline in storage costs (per gigabyte) and exponential growth in distributed cloud compute (GPUs, TPUs, distributed RAM).
* **Algorithmic Maturation:** Breakthroughs in gradient boosting (XGBoost, LightGBM), deep learning architectures, and scalable nearest-neighbor algorithms.

---

## 2. Key Definition Glossary

* **Data Mining:** The automated extraction of hidden, previously unknown, and actionable predictive patterns from large datasets.
* **Knowledge Discovery in Databases (KDD):** The overarching multi-step process comprising Data Selection, Data Preprocessing, Data Transformation, Data Mining, and Pattern Evaluation/Interpretation.
* **Curse of Dimensionality:** A mathematical phenomenon where as the number of features (dimensions $D$) grows, the volume of feature space increases exponentially, causing data points to become sparse and Euclidean distances between points to become indistinguishable.

---

## 3. Applied Scenarios & Failure Modes

### Scenario: High Dimensionality & Overfitting in Clinical Datasets
* **Industrial Scenario:** An analyst builds a predictive model for hospital readmissions with 10,000 patient records and 4,000 one-hot encoded diagnosis codes without dimensionality reduction or regularization.
* **Failure Mode:** The model suffers severely from the Curse of Dimensionality. It achieves 99% training accuracy by memorizing noise but drops to 52% on test data (severe overfitting).
* **Remediation:** Apply feature selection (VarianceThreshold, Mutual Information) or dimensionality reduction (PCA / TruncatedSVD) and group granular ICD-9 codes into higher-level clinical categories.

---

# MODULE 2: The Four Types of Data Analytics

## 1. Academic Context & Theoretical Deep Dive

Analytics is categorized along two axes: **Value Generated** (Y-axis) versus **Complexity / Autonomy** (X-axis). Moving from hindsight to foresight increases strategic leverage:

```
VALUE
  ▲                                                            [PRESCRIPTIVE]
  │                                                      "What should we do?"
  │                                                      (Optimization / Sim)
  │                                          [PREDICTIVE]
  │                                    "What will happen?"
  │                                    (ML / Forecasting)
  │                        [DIAGNOSTIC]
  │                  "Why did it happen?"
  │                  (Root Cause / Drilldown)
  │        [DESCRIPTIVE]
  │  "What happened?"
  │  (Reports / Dashboards)
  └────────────────────────────────────────────────────────────────────────► COMPLEXITY
```

### The Doctor Analogy (The Definitive Pedagogical Framework)
To master and defend the four types of analytics in oral examinations, apply **The Clinical Physician Framework**:
* **Descriptive Analytics (*"What happened?"*):** The nurse places a thermometer in the patient's mouth. The thermometer reads $39.5^\circ\text{C}$ (Fever). Past-tense factual summary.
* **Diagnostic Analytics (*"Why did it happen?"*):** The doctor orders blood cultures and chest X-rays. The lab results confirm a high white blood cell count and bacterial consolidation in the left lung (Bacterial Pneumonia). Root cause identification.
* **Predictive Analytics (*"What is likely to happen?"*):** Based on the patient's age (72) and comorbid diabetes, the risk prediction algorithm calculates an 82% probability of developing septic shock within 36 hours if untreated. Probabilistic foresight.
* **Prescriptive Analytics (*"What should we do?"*):** The clinical decision engine recommends an exact dosage: *"Administer 1.5g IV Vancomycin immediately, initiate 30 mL/kg saline fluid resuscitation, and repeat lactate panel in 2 hours."* Specific actionable optimization.

---

## 2. Key Definition Glossary

* **Descriptive Analytics:** The condensation of raw historical data into human-interpretable summaries, KPI metrics, and dashboards (e.g., mean readmission rate, total revenue).
* **Diagnostic Analytics:** The investigative drill-down into historical data to identify correlations, causal drivers, and anomalies explaining observed performance.
* **Predictive Analytics:** The application of statistical models and machine learning algorithms to historical data to forecast future probabilities or numerical trajectories.
* **Prescriptive Analytics:** The application of mathematical optimization, simulation models, and heuristics to evaluate potential decisions and automatically prescribe the optimal course of action.

---

# MODULE 3: The Eight Core Data Mining Techniques

## 1. Academic Context & Theoretical Deep Dive

Data mining algorithms are broadly bifurcated into **Supervised** (learning with ground-truth targets) and **Unsupervised** (discovering intrinsic structures without target labels):

| Technique | Learning Paradigm | Target Variable Type | Core Objective | Representative Algorithms |
| :--- | :--- | :--- | :--- | :--- |
| **1. Classification** | Supervised | Categorical / Discrete | Predict discrete class labels | Decision Tree, Random Forest, XGBoost, Logistic Reg, SVM |
| **2. Regression** | Supervised | Continuous Numerical | Predict numerical values | OLS Linear Regression, Ridge, Lasso, SVR |
| **3. Clustering** | Unsupervised | None | Group instances by geometric proximity | K-Means, DBSCAN, Agglomerative Hierarchical |
| **4. Association Rules** | Unsupervised | Transactional Baskets | Discover item co-occurrence rules | Apriori, FP-Growth, ECLAT |
| **5. Anomaly Detection**| Semi/Unsupervised | Binary (Normal / Outlier)| Flag rare, abnormal instances | Isolation Forest, Local Outlier Factor (LOF), One-Class SVM |
| **6. Sequential Patterns**| Unsupervised | Time-Ordered Sequences| Discover time-dependent sequences | Generalized Sequential Patterns (GSP), PrefixSpan |
| **7. Text Mining / NLP** | Supervised / Unsupervised | Text Strings | Extract structured signals from text | TF-IDF, Word2Vec, LDA Topic Modeling, VADER |
| **8. Dimensionality Red.**| Unsupervised | Continuous Vectors | Compress dimensions while preserving variance | PCA, t-SNE, UMAP, TruncatedSVD |

### Association Rule Mining Mathematics: Support, Confidence, and Lift
For an association rule $A \Rightarrow B$ (e.g. $\{\text{Insulin}\} \Rightarrow \{\text{Metformin}\}$):
* **Support:** Frequency of both items occurring together:
  $$\text{Support}(A \Rightarrow B) = P(A \cap B) = \frac{\text{Count}(A \cap B)}{N}$$
* **Confidence:** Conditional probability that $B$ is purchased given that $A$ is purchased:
  $$\text{Confidence}(A \Rightarrow B) = P(B \mid A) = \frac{P(A \cap B)}{P(A)} = \frac{\text{Count}(A \cap B)}{\text{Count}(A)}$$
* **Lift:** The ratio of observed joint occurrence to expected occurrence under independence:
  $$\text{Lift}(A \Rightarrow B) = \frac{P(A \cap B)}{P(A) \cdot P(B)} = \frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)}$$
  * $\text{Lift} = 1.0$: $A$ and $B$ are completely independent.
  * $\text{Lift} > 1.0$: Positive association (buying $A$ increases likelihood of buying $B$).
  * $\text{Lift} < 1.0$: Negative association (substitutes).

---

# MODULE 4: Structured Data Preprocessing & Exploratory Data Analysis (EDA)

## 1. Academic Context & Theoretical Deep Dive

### 1.1 The Four Data Measurement Scales (Stevens' Taxonomy)
* **Nominal:** Qualitative categories with no intrinsic mathematical order. Operations: Mode, Count, Equality ($=$ or $\neq$). Examples: Blood Type (A, B, AB, O), Hospital Ward, Gender.
* **Ordinal:** Categorical variables with a meaningful rank or hierarchy, but unequal or unquantifiable distances between ranks. Operations: Median, Percentiles, Ordering ($<$ or $>$). Examples: Cancer Stage (Stage I, II, III, IV), Patient Acuity (Low, Medium, High).
* **Interval:** Quantitative numerical scale with equal distances between units, but **no true zero point** (zero does not mean absence of the attribute). Operations: Mean, Addition, Subtraction. Examples: Temperature in Celsius ($0^\circ\text{C}$ is not the absence of heat), Calendar Year.
* **Ratio:** Quantitative numerical scale with equal intervals and an **absolute, meaningful zero point** (zero means complete absence). Operations: Mean, Multiplication, Division, Ratios. Examples: Patient Age, Length of Stay, Blood Glucose ($0\text{ mg/dL}$ represents total absence), Medication Count.

### 1.2 Outlier Detection: The 1.5 $\times$ IQR Rule
The Interquartile Range (IQR) represents the middle 50% of sorted continuous data:
$$\text{IQR} = Q_3 - Q_1$$
* **Inner Fences (Mild Outliers):**
  $$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$
* **Outer Fences (Extreme Outliers):**
  $$\text{Lower Outer} = Q_1 - 3.0 \times \text{IQR}, \quad \text{Upper Outer} = Q_3 + 3.0 \times \text{IQR}$$
* **Outlier Treatment Decision Rules:**
  * *Trimming / Removal:* Use ONLY when outliers are verified data entry corruptions or measurement errors.
  * *Winsorization (Capping):* Replace values exceeding upper/lower fences with the fence values ($Q_3 + 1.5\text{IQR}$ or 99th percentile). Preserves sample size.
  * *Robust Scaling:* Keep values as-is and apply RobustScaler when modeling with tree algorithms (Random Forest, XGBoost) that are inherently invariant to monotonic feature scale.

### 1.3 Missing Data Mechanisms: MCAR vs. MAR vs. MNAR
Donald Rubin's missing data taxonomy defines the probability of missingness $P(M \mid Y_{obs}, Y_{mis})$:
* **Missing Completely at Random (MCAR):**
  $$P(M \mid Y_{obs}, Y_{mis}) = P(M)$$
  The probability of missingness is completely independent of both observed and unobserved data. Example: A blood tube drops and shatters on the floor randomly.
  * *Diagnostic Test:* **Little's MCAR Test**. If $p > 0.05$, fail to reject the null hypothesis $\rightarrow$ data is consistent with MCAR.
* **Missing at Random (MAR):**
  $$P(M \mid Y_{obs}, Y_{mis}) = P(M \mid Y_{obs})$$
  The probability of missingness depends on other observed patient features, but not on the missing value itself. Example: Younger patients (observed age) are less likely to have blood pressure records recorded because doctors skip routine checks on the young.
  * *Diagnostic Test:* Column-by-column Chi-Square / t-tests comparing missingness indicators against other observed variables.
* **Missing Not at Random (MNAR):**
  $$P(M \mid Y_{obs}, Y_{mis}) \text{ depends on } Y_{mis}$$
  Missingness directly depends on the unobserved value itself. Example: Severely depressed patients deliberately refuse to answer depression questionnaire items.
  * *Remediation:* Create an explicit missing indicator flag (`is_missing = 1`) so models can learn the informativeness of non-response.

### 1.4 Imputation Strategy Decision Rules
1. **Missingness Threshold:** If a column has $> 40\%–50\%$ missing values (e.g. `weight` in the Diabetes dataset has $97\%$ missingness), drop the entire column unless domain rationale dictates creating an indicator.
2. **MCAR / Numerical:** Mean imputation (if normally distributed, skewness $\approx 0$) or Median imputation (if skewed or outliers present).
3. **MAR / Multivariate Numerical:** Iterative Imputer (MICE - Multivariate Imputation by Chained Equations) or KNN Imputer ($k=5$).
4. **Categorical Variables:** Mode imputation or encode missingness as an explicit category: `"Unknown"` / `"Missing"`.

### 1.5 Feature Scaling Decision Matrix
* **Min-Max Normalization:**
  $$x' = \frac{x - x_{min}}{x_{max} - x_{min}} \in [0, 1]$$
  * *Use When:* Features must be bounded strictly to $[0, 1]$ (e.g. image pixel values, neural network activations). Highly sensitive to outliers!
* **Z-Score Standardization:**
  $$z = \frac{x - \mu}{\sigma}$$
  * *Use When:* Algorithms assume zero-centered data with unit variance (PCA, Logistic Regression, Linear Regression, SVM). Outliers are preserved.
* **Robust Scaler:**
  $$x' = \frac{x - \text{median}}{\text{IQR}}$$
  * *Use When:* Data contains severe outliers that would distort mean $\mu$ and standard deviation $\sigma$.

### 1.6 Hypothesis Testing & The LINE + M Framework
Before running parametric statistical tests (Linear Regression, ANOVA, t-test), audit test assumptions using **LINE + M**:
* **L — Linearity:** Bivariate scatter plots show linear relationships.
* **I — Independence:** Durbin-Watson statistic $\approx 2.0$ (no autocorrelation in residuals).
* **N — Normality:** Residuals follow Gaussian distribution. Audited via **Shapiro-Wilk test** ($p > 0.05$) and Q-Q plots.
* **E — Equal Variance (Homoscedasticity):** Residual variance is uniform across predicted values. Audited via **Levene's test** or Breusch-Pagan test ($p > 0.05$).
* **M — Multicollinearity:** Independent predictors are not strongly correlated. Audited via **Variance Inflation Factor (VIF)**. If $\text{VIF} > 5.0$, severe collinearity exists.

---

# MODULE 5: Unstructured Data Preprocessing & Clinical Notes NLP

## 1. Academic Context & Theoretical Deep Dive

### 1.1 The 10-Step NLP Preprocessing Pipeline
Text data in EHR clinical notes or customer reviews is unstructured and noisy. Transforming raw text into model-ready features follows a rigid 10-step sequence:
1. **Case Normalization:** Convert all characters to lowercase to collapse identical words (`"Patient"` $\rightarrow$ `"patient"`).
2. **Contraction Expansion:** Expand conversational contractions (`"can't"` $\rightarrow$ `"cannot"`).
3. **HTML / Noise Cleaning:** Strip markup tags (`<br>`, `<div>`) and non-ASCII artifacts using regex.
4. **Punctuation & Special Character Removal:** Strip symbols (`!@#$%^&*()`) while preserving decimal digits if clinically relevant.
5. **Tokenization:** Split text strings into atomic word tokens using whitespace or NLTK's `word_tokenize`.
6. **Stopword Removal:** Remove ubiquitous, uninformative grammatical words (`"the"`, `"is"`, `"at"`, `"which"`).
7. **Stemming vs. Lemmatization:**
   * *Stemming (Porter / Snowball):* Crude heuristic chopping of word suffixes. Fast but produces non-dictionary stems (`"studies"` $\rightarrow$ `"studi"`, `"caring"` $\rightarrow$ `"car"`).
   * *Lemmatization (WordNet):* Uses morphological dictionary analysis and Part-of-Speech (POS) tags to reduce words to their true lemma (`"better"` $\rightarrow$ `"good"`, `"running"` as verb $\rightarrow$ `"run"`).
8. **N-Gram Generation:** Extract contiguous sequences of $n$ items (Unigrams $n=1$, Bigrams $n=2$, Trigrams $n=3$) to capture contextual phrases like `"myocardial infarction"` or `"blood pressure"`.
9. **Vectorization (BoW vs. TF-IDF):** Convert token frequencies into numerical vectors.
10. **Topic Modeling / Sentiment Scoring:** Extract thematic topic distributions (LDA) or polarity scores.

### 1.2 Vectorization: Bag of Words vs. TF-IDF
* **Bag of Words (CountVectorizer):** Simply counts token occurrences $c(t, d)$ in document $d$. Overweights common words that appear across all documents.
* **TF-IDF (Term Frequency-Inverse Document Frequency):**
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
  $$\text{IDF}(t, D) = \log\left(\frac{N}{|\{d \in D : t \in d\}| + 1}\right)$$
  * $N$: Total number of documents in corpus.
  * Term $t$ receives high TF-IDF if it appears frequently in document $d$, but appears rarely across the general corpus $D$.

### 1.3 Latent Dirichlet Allocation (LDA) Topic Modeling
LDA is an unsupervised probabilistic generative model that assumes:
* Every document $d$ is modeled as a multinomial distribution over $K$ latent topics:
  $$\theta_d \sim \text{Dirichlet}(\alpha)$$
* Every topic $k$ is modeled as a multinomial distribution over the vocabulary $V$:
  $$\phi_k \sim \text{Dirichlet}(\beta)$$
* **Hyperparameters:**
  * $\alpha$: Prior on document-topic density. High $\alpha \rightarrow$ documents contain mixtures of many topics; low $\alpha \rightarrow$ documents focus on 1–2 topics.
  * $\beta$: Prior on topic-word density. High $\beta \rightarrow$ topics cover many words; low $\beta \rightarrow$ topics contain distinct, focused vocabularies.
* **Evaluation:**
  * *Perplexity:* Measures model surprise on held-out text (lower is better).
  * *Coherence Score ($C_v$):* Measures semantic similarity of top words in a topic based on sliding-window co-occurrence (higher is better, typical target $C_v > 0.55$).

---

# MODULE 6: Master Formula Cheat Sheet & Decision Matrices

## 1. Formulas Quick Reference Table

| Formula / Metric | Mathematical Expression | Key Interpretation / Rule |
| :--- | :--- | :--- |
| **Interquartile Range** | $\text{IQR} = Q_3 - Q_1$ | Spread of the middle 50% of sorted continuous data |
| **Outlier Lower Fence** | $Q_1 - 1.5 \times \text{IQR}$ | Observations below this fence are mild outliers |
| **Outlier Upper Fence** | $Q_3 + 1.5 \times \text{IQR}$ | Observations above this fence are mild outliers |
| **Min-Max Scaling** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Rescales feature strictly to $[0, 1]$; sensitive to outliers |
| **Z-Score Scaling** | $z = \frac{x - \mu}{\sigma}$ | Standardizes to $\mu=0, \sigma=1$; assumes Gaussian dist |
| **Robust Scaling** | $x' = \frac{x - \text{median}}{\text{IQR}}$ | Rescales using median and IQR; resilient to severe outliers |
| **Association Support** | $P(A \cap B) = \frac{\text{Count}(A \cap B)}{N}$ | Frequency of joint itemset occurrence |
| **Association Confidence** | $P(B \mid A) = \frac{\text{Count}(A \cap B)}{\text{Count}(A)}$ | Reliability of the directional rule $A \Rightarrow B$ |
| **Association Lift** | $\frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)}$ | Lift $> 1.0$ indicates positive correlation; $=1.0$ independent |
| **TF-IDF Weight** | $\text{TF} \times \log\left(\frac{N}{\text{DF} + 1}\right)$ | Penalizes ubiquitous words, rewards distinct domain tokens |

---

# MODULE 7: Master Identification Quiz & Diagnostic Defense Q&As

## 1. High-Yield Identification Questions (Modules 1–5)

### Q1: What is the formal definition of Data Mining according to Fayyad et al. (1996)?
> **Answer:** The non-trivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns from massive, heterogeneous data repositories.

### Q2: What type of analytics answers the question: "Why did hospital readmissions spike in December 2024?"
> **Answer:** **Diagnostic Analytics** (drills down into historical data to identify root causes and anomalous drivers).

### Q3: A patient's temperature is recorded as 38.5°C. What level of data measurement scale does Celsius belong to, and why?
> **Answer:** **Interval Scale**, because temperature in Celsius has equal, standardized units of measurement, but lacks an absolute true zero point ($0^\circ\text{C}$ is arbitrarily calibrated to the freezing point of water, not the total absence of thermodynamic heat).

### Q4: In an audit of a hospital dataset, a feature exhibits $Q_1 = 12$ days and $Q_3 = 24$ days for Length of Stay. What are the exact lower and upper outlier fences under the $1.5 \times \text{IQR}$ rule?
> **Answer:**
> * $\text{IQR} = 24 - 12 = 12\text{ days}$
> * $\text{Lower Fence} = 12 - 1.5(12) = 12 - 18 = -6\text{ days}$ (bounded to 0 in practice)
> * $\text{Upper Fence} = 24 + 1.5(12) = 24 + 18 = 42\text{ days}$

### Q5: What statistical test is used to evaluate whether an entire tabular dataset's missingness conforms to Missing Completely at Random (MCAR)?
> **Answer:** **Little's MCAR Test** (evaluates mean differences across all missingness patterns simultaneously; $p > 0.05$ indicates consistency with MCAR).

### Q6: When one-hot encoding a categorical feature with $k=5$ unique categories, why must one dummy column be dropped when training an Ordinary Least Squares (OLS) regression model?
> **Answer:** To prevent the **Dummy Variable Trap (Perfect Multicollinearity)**. If all $k$ indicators are included alongside an intercept, the sum of the columns equals the intercept vector ($\sum x_i = 1$), causing the design matrix $X^T X$ to be singular and non-invertible.

### Q7: Compare Stemming and Lemmatization in NLP. Which one produces `"studi"` from `"studying"` and which one produces `"study"`?
> **Answer:** **Stemming** uses heuristic suffix chopping and produces `"studi"` (a non-dictionary root). **Lemmatization** uses vocabulary dictionaries and morphological context to produce `"study"` (a valid lemma).

### Q8: What does a Lift score of $2.4$ indicate for the association rule $\{\text{Metformin}\} \Rightarrow \{\text{Glipizide}\}$?
> **Answer:** Patients prescribed Metformin are **$2.4$ times more likely** to also be prescribed Glipizide than would be expected if the two medications were prescribed completely independently.

---
*DS312 Master Companion Guide compiled for academic review and laboratory defense mastery.*
