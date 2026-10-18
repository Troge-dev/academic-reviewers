# The Philosophy of Looking: How Topic 01 Cured My "Dataset-Hopping" Panic

> **Subtitle / Hero Deck:**  
> *"Before DS311, I rushed straight into models and abandoned messy datasets out of panic. Then Ma'am Chris walked us through John Tukey and Francis Anscombe in Topic 01 — teaching me that data must be seen, not just calculated, and why the messy surprises are where the real story lives."*

---

## Metadata & Byline
* **Author:** Rogelio Mandamian (BS Data Science, Section DS3A)
* **Institution:** University of Science and Technology of Southern Philippines (USTP)
* **Course:** DS311: Exploratory Data Analysis
* **Instructor:** Ma'am Chris Jallaine S. Mugot
* **Topic Focus:** Topic 01 — Introduction to Exploratory Data Analysis
* **Core Inspiration Aesthetic:** Minimalist Baker (Uncluttered, simple, warm editorial typography)

---

## Table of Contents & Navigation
1. [The Confession: My Backward "Confirmatory" Rush](#1-the-confession-my-backward-confirmatory-rush)
2. [The "Dataset-Hopping" Panic & Stumbling Upon Egypt](#2-the-dataset-hopping-panic--stumbling-upon-egypt)
3. [When Numbers Mislead: Francis Anscombe's 1973 Wake-Up Call](#3-when-numbers-mislead-francis-anscombes-1973-wake-up-call)
4. [The Four R's of EDA: John Tukey's Core Philosophy](#4-the-four-rs-of-eda-john-tukeys-core-philosophy)
5. [Interactive Lab: See Anscombe & Leverage in Real Time](#5-interactive-lab-see-anscombe--leverage-in-real-time)
6. [A Cycle of Questions: How I Actually Walk Through the 7 Steps](#6-a-cycle-of-questions-how-i-actually-walk-through-the-7-steps)
7. [From Foundations to Practice: My Academic Project Archive](#7-from-foundations-to-practice-my-academic-project-archive)
8. [The Minimalist Data Manifesto (Topic 01 Edition)](#8-the-minimalist-data-manifesto-topic-01-edition)
9. [Curated Inspiration & Recommended Reading](#9-curated-inspiration--recommended-reading)

---

## 1. The Confession: My Backward "Confirmatory" Rush

Let me share an honest confession about how I used to approach data science before entering DS311: I was doing almost everything completely backward.

Whenever our professors gave us an analytical assignment, my immediate instinct was to formulate a conclusion in my head first, load the dataset, and rush to ask: *"Does this confirm what I want to prove?"* I would skip the foundational legwork and jump straight into multivariate scatter plots, high-dimensional projections, and correlation heatmaps without spending even five minutes inspecting what the individual columns actually contained.

I didn't pay attention to whether a feature had a severe right skew, whether the variance was dominated by a couple of massive outliers, or what the units of measurement even meant. To me, a dataset was just a monolithic block of numbers that I wanted to shove into an algorithm as fast as possible.

Inevitably, reality would push back hard. I would find a column with 30% null values. A variable would have negative values where only positive numbers made sense. Or a correlation coefficient would read as a weak $r = 0.12$ when I was convinced a massive relationship existed.

And whenever I hit those roadblocks, I panicked. I had memorized textbook terms like "mean imputation," "median imputation," and "forward fill," but in practice, I had little to no idea which imputation I should actually choose or what damage it would cause. I treated missing values like an annoying bug to be erased — blindly writing `df.fillna(df.mean())` just to stop my code from crashing. When the resulting models fell apart or produced nonsensical conclusions, I fell into the most toxic habit of all: **chronic dataset-hopping**.

> *"It is very common to spend the most time in the exploratory data analysis stage in order to have a well-made output and to prevent future inconvenience."*  
> — **Ma'am Chris Jallaine S. Mugot**, DS311 Instructor

---

## 2. The "Dataset-Hopping" Panic & Stumbling Upon Egypt

During our 2nd-year **DATAVIS** course, my group and I hit this wall repeatedly. We would spend hours scouting Kaggle for datasets with exciting titles, load them into Google Colab, encounter missing values and bizarre percentages, enter a cold sweat at 1:30 AM, and abandon the project to hunt for something easier.

We hopped through multiple dead ends until we finally stumbled upon the **Egypt Population Density dataset**.

Unlike the chaotic datasets that had triggered our panic, the Egypt dataset wasn't really a problem. It provided coherent, structured records of Egypt's governorates, total populations, and inhabited land areas. Stumbling upon it felt like striking gold: it finally gave us a reliable foundation to map the dramatic demographic contrast between the hyper-dense Nile Valley corridor (with Cairo and Giza packing millions into tight urban spaces) and the vast, sparsely populated desert frontier governorates.

### Associated Visual Assets
* **Figure 1:** `assets/images/contents/egypt_raw_flat_view.png`  
  *Caption:* The Raw Flat View — Exploring Egypt's governorate distributions, populations, and inhabited land areas stumbled upon during our DATAVIS project.
* **Figure 2:** `assets/images/contents/Egypt_Population_Density_Map_Preview_web.jpg` (Full-res: `assets/images/contents/Egypt_Population_Density_Map_Preview.png`)  
  *Caption:* The Final DATAVIS Map — Visualizing Egypt's acute population concentration along the Nile corridor (300 DPI high-resolution export).

### The Realization
The Egypt dataset allowed us to deliver an infographic map we were genuinely proud of. But here was the honest realization that hit me later when taking DS311:  
**In DATAVIS, I had to get lucky by hopping around until I stumbled upon a dataset that didn't scare me.**

Finding a neat dataset let me dodge the hard questions. But in the real world — and in professional data science — you cannot simply abandon a company's data or a research project because the columns are messy or skewed. Running away from messy data is running away from data science itself.

This became undeniable in our **DS311 Laboratory 1 (Water Potability Data Cleaning)**. When auditing 3,276 water samples across aquatic chemical parameters (pH, Sulfate, Trihalomethanes), we were confronted with selective laboratory test omissions. Expensive gas chromatography tests weren't performed at every rural station — resulting in missing records. If I had naively executed `fillna(mean)` like I used to in DATAVIS, I would have manufactured fake chemical clusters and falsified drinking water safety!

---

## 3. When Numbers Mislead: Francis Anscombe's 1973 Wake-Up Call

The complete turning point occurred in our very first week of **DS311**, when our instructor, **Ma'am Chris Jallaine S. Mugot**, walked us through **Topic 01: Introduction to Exploratory Data Analysis**.

Right at the beginning of the notebook, Ma'am Chris introduced us to **Anscombe's Quartet** — a collection of four synthetic datasets published in 1973 by statistician Francis Anscombe in *The American Statistician*. His explicit purpose was to counter the prevailing attitude in statistics that numerical summaries are sufficient and that graphical analysis is merely informal or decorative.

Each of the four datasets contains 11 coordinate pairs $(x, y)$. When you calculate their standard summary statistics, you get values that are nearly identical across all four:

### Summary Statistics Across All 4 Datasets
| Metric | Dataset I | Dataset II | Dataset III | Dataset IV |
| :--- | :--- | :--- | :--- | :--- |
| **Number of Points ($N$)** | 11 | 11 | 11 | 11 |
| **Mean of X ($\mu_X$)** | 9.00 | 9.00 | 9.00 | 9.00 |
| **Mean of Y ($\mu_Y$)** | 7.50 | 7.50 | 7.50 | 7.50 |
| **Sample Variance of X ($s^2_X$)** | 11.00 | 11.00 | 11.00 | 11.00 |
| **Sample Variance of Y ($s^2_Y$)** | 4.125 | 4.125 | 4.125 | 4.125 |
| **Pearson Correlation ($r$)** | 0.816 | 0.816 | 0.816 | 0.816 |
| **OLS Regression Line** | $y = 3.00 + 0.50x$ | $y = 3.00 + 0.50x$ | $y = 3.00 + 0.50x$ | $y = 3.00 + 0.50x$ |
| **Coefficient of Determination ($R^2$)**| 66.6% | 66.6% | 66.6% | 66.6% |

Before DS311, I would have looked at this table and concluded without hesitation: *"These four datasets describe the exact same phenomenon."*

Then Ma'am Chris showed us the plots:
* **Figure 3:** `assets/images/charts/anscombes_quartet_proof.svg`  
  *Caption:* Anscombe's Quartet — Four wildly different geometries disguised behind the exact same summary numbers.

Seeing those plots was a physical shock. The four structures could not be more radically different:
1. **Dataset I: The "Normal" Case** — A clean, linear relationship with natural scatter. This is the only dataset of the four where fitting an Ordinary Least Squares (OLS) line is genuinely honest and appropriate.
2. **Dataset II: The Smooth Quadratic Arc** — The data is not random noise at all — it follows a deterministic curved parabola. Fitting a straight line here is systematically wrong, yet no summary statistic would ever reveal that without a plot.
3. **Dataset III: Straight Line Hijacked by an Outlier** — Except for one single rogue observation, this dataset is a strict, perfect collinear line. That single vertical outlier is the entire story of the dataset, yet it is completely invisible in the numerical summary.
4. **Dataset IV: The Extreme Leverage Hostage** — All ten primary observations share the exact same X value ($x = 8.0$). The regression line, slope, and correlation are entirely manufactured by a single leverage point way out at $x = 19.0$!

> *"Data must be seen, not just calculated. Anscombe's Quartet is a proof that you cannot understand data by summarizing it alone — and EDA is the discipline built around that exact lesson."*

---

## 4. The Four R's of EDA: John Tukey's Core Philosophy

Who invented this discipline? In 1977, statistician **John Tukey** published his foundational book *Exploratory Data Analysis*. Tukey argued that data analysis should never begin with rigid assumptions or predefined hypotheses.

Instead of immediately asking, *"Does this confirm my hypothesis?"*, EDA asks, **"What is this data trying to tell me?"** This fundamental shift in perspective is what distinguishes exploratory work from traditional confirmatory statistics.

As Ma'am Chris taught us in Topic 01, Tukey encapsulated this philosophy into **The Four R's**, which I now treat as my personal data science compass:

### R1: Resistance (Median Over Mean)
* **Core Idea:** In my coursework, I learned why Ma'am Chris emphasized median over mean. When working with variables like income, hospital lengths of stay, or municipal densities where extreme observations skew the scale, the mean gets pulled by outliers, but the median stays anchored to where the bulk of real life lives.
* **My Takeaway:** *My findings must reflect the bulk of the data, not be hijacked by its exceptions.*

### R2: Residuals (Data = Fit + Residual)
* **Core Idea:** I used to only glance at $R^2$ and feel satisfied when a model fit. But Tukey taught me that the residual — what the model failed to explain — is often the most interesting part. On curved data like Anscombe II, inspecting the residuals immediately exposes the parabola that the straight regression line missed.
* **My Takeaway:** *Never be satisfied with a pattern that fits. Always inspect what was left behind.*

### R3: Re-expression (Transform to Clarify)
* **Core Idea:** The raw form a dataset arrives in is rarely the best form for understanding it. When I worked with the Egypt dataset, Cairo and Giza squashed the rest of the country flat against zero. Taking $\log_{10}(	ext{density})$ compressed the heavy tail and revealed the real bimodal distribution between oasis frontiers and Nile metros.
* **My Takeaway:** *Transformations aren't cheating; they are the magnifying glass that reveals reality.*

### R4: Revelation (Make Invisible Visible)
* **Core Idea:** The ultimate purpose of EDA is to reveal what is hidden. Tukey championed visual plots because a well-crafted graphic shows in seconds what pages of summary tables conceal. Resistance protects my findings from distortion, Residuals show what I missed, and Re-expression clears the path.
* **My Takeaway:** *If my analysis hasn't shown me something I didn't already know, I haven't finished exploring.*

### Connection: How Re-expression Explained the Egypt Dataset
This was the huge "aha!" moment connecting my DATAVIS experience to DS311. In the Egypt Population Density dataset, Cairo and Giza had astronomical population densities (>15,000 people/km²), while desert governorates like New Valley and Matruh had less than 5 people/km².

Plotted on a raw scale, 90% of the governorates were squashed flat against the zero axis. Back in 2nd year, I thought the data was "broken." But Tukey's third R taught me: **Re-expression!** By applying a simple $\log_{10}(	ext{density})$ transform, the long right tail was compressed, revealing the beautiful bimodal demographic reality of metropolitan clusters vs oasis frontiers. The data wasn't broken — it was simply waiting for the right lens!

---

## 5. Interactive Lab: See Anscombe & Leverage in Real Time

To experience Francis Anscombe's 1973 discovery for yourself, I built an interactive 2D regression laboratory. You can select presets to simulate Anscombe's four classic scenarios, or click anywhere inside the canvas to drop points in real time and watch how Pearson's $r$, OLS slope ($eta_1$), and $R^2$ react dynamically.

*Notice what happens when you switch between the presets:* Even when the summary statistics match ($r pprox 0.816$, $eta_1 pprox 0.500$, $R^2 pprox 66.6\%$), the physical story on the canvas changes completely. In **Anscombe IV (Leverage)**, try clicking and dropping points near the left vertical cluster — notice how stubbornly the line remains pinned to that single lonely dot on the far right. That is **influence in action**.

---

## 6. A Cycle of Questions: How I Actually Walk Through the 7 Steps

In traditional mathematics and introductory statistics classes, I was conditioned to follow rigid, step-by-step formulas: plug in the numbers, calculate the result, and get the single right answer.

In Topic 01, Ma'am Chris taught us that **Exploratory Data Analysis does not work that way**:

> *"EDA has no strict rules and is iterative. It is not governed by a universal checklist that must be followed in the same order every time. Instead, it is shaped by the data itself and by the questions that emerge during analysis."*

Today, instead of following a mindless checklist, I navigate my projects using these **7 Common Practitioner Steps** as an iterative compass based on what I have learned through trial, error, and real laboratory headaches:

1. **Understand Context and the Data Source:**  
   In my earlier DATAVIS projects, I used to ignore where data came from entirely — I would just download a CSV from Kaggle and immediately start typing code. That ignorance burned me repeatedly because I had no idea what the numbers actually tracked. When working on our **CDO Health Access project**, I learned that knowing whether healthcare clinic coordinates came from OpenStreetMap volunteers versus official DOH or CBMS censuses determined what the records truly meant before I wrote a single line of code. Data never exists in a vacuum; understanding the source protects me from drawing confident, embarrassing conclusions on misunderstood columns.

2. **Inspect the Structure of the Data:**  
   My old bad habit was jumping straight into plotting charts without checking the data's skeleton first. I would run into bizarre syntax crashes halfway through an analysis simply because a numeric column was formatted as an `object` string, or because the dataset had thousands of hidden duplicate rows. Now, the moment I load any dataframe into Jupyter, my mandatory first reflex is running `df.shape` and `df.info()`. It is like walking into a room and turning on the lights before touching anything on the shelves.

3. **Assess Data Quality:**  
   This was my biggest wake-up call during our **Lab 01 Water Potability activity** and our **Diabetes 130-US Hospitals dataset**. In 2nd year, I treated missing values like a nuisance to be wiped with `fillna(mean)`. But when auditing 3,276 water samples, I discovered that missing Sulfate and Trihalomethanes readings weren't random errors — they were selective laboratory test omissions because rural stations lacked expensive chromatography equipment. And in the diabetes data, weight was missing in 97% of patient records! I learned that you cannot trust any conclusion built on unexamined data. Now, I document and understand what the gaps actually mean instead of rushing to cover them up.

4. **Examine Individual Variables (Univariate):**  
   Before Ma'am Chris taught us Topic 01 and Topic 05, I completely ignored single variables — I wanted to see multi-variable interactions right away. But when I audited the **Egypt Population Density dataset**, looking at population density without inspecting the distribution squashed everything flat against zero because Cairo and Giza were massive outliers. Now, I audit every column individually first: I check the 5-number summary and compare the median against the mean to spot skewness and leverage before they ruin downstream modeling.

5. **Examine Relationships Between Variables (Bivariate & Lenses):**  
   When I reached **Lab 04 (NYC Yellow Taxi Trips)** and paired trip distance with fare amount, bivariate inspection became real. A single scalar like Pearson's $r$ would have tricked me into believing a simple linear trend existed, but plotting the scatter revealed flat-rate airport trips where 0 miles cost $50, and GPS glitches where 150 miles cost $10! In our **CDO Health Access research**, examining travel times against socioeconomic vulnerability revealed spatial divides that no single variable could show. I learned that Univariate, Bivariate, and Multivariate are lenses, not locks — I move between them as my understanding matures.

6. **Identify Patterns, Anomalies, and Surprises:**  
   This is the core orientation Ma'am Chris instilled in me. Instead of passively writing code, I constantly ask: *"Does this number make physical sense in the real world? If not, why?"* When I saw 0-mile taxi trips with $50 fares in Lab 04, or rural barangays with zero primary health facilities in our CDO spatial study, those weren't errors to throw away — they were the real story! Finding the anomaly is where genuine discovery begins.

7. **Formulate Questions and Hypotheses:**  
   I used to think that the goal of data analysis was to finish with an undisputed, final answer on day one. But as Ma'am Chris reminded us, EDA doesn't give you final answers — it gives you better, sharper questions. When I finish exploring a dataset now, I walk away with informed, testable hypotheses: *"Why do upland barangays show acute travel-time impedance?"* or *"Does patient readmission spike when HbA1c isn't tested during inpatient admission?"* EDA transforms my blind curiosity into rigorous science.

---

## 7. From Foundations to Practice: My Academic Project Archive

The lessons from Topic 01 don't stay confined to a single notebook. They form the bedrock of how I now approach every data science problem in my 3rd-year coursework:

* **Project 1: Water Potability Data Cleaning Pipeline (DS311 Lab 01)**  
  *URL:* [https://github.com/Troge-dev/Water-Potability-Data-Cleaning](https://github.com/Troge-dev/Water-Potability-Data-Cleaning)  
  *Description:* Auditing 3,276 aquatic samples across 10 physicochemical parameters. Demonstrating why selective laboratory omissions require domain-grounded imputation rather than blind statistical fills.
* **Project 2: Socioeconomic Vulnerability & Health Access in CDO (DS312)**  
  *URL:* [https://github.com/Troge-dev/Socioeconomic-Vulnerability-Health-Access-CDO](https://github.com/Troge-dev/Socioeconomic-Vulnerability-Health-Access-CDO)  
  *Description:* Spatial accessibility modeling across Cagayan de Oro City's 80 barangays using Enhanced 2SFCA travel times, correlating healthcare deficits with socioeconomic deprivation indices.
* **Project 3: Diabetes 130-US Hospitals 30-Day Readmission (DS312)**  
  *URL:* [https://github.com/Troge-dev/Diabetes-130-US-Hospitals-30-Day-Readmission-Prediction](https://github.com/Troge-dev/Diabetes-130-US-Hospitals-30-Day-Readmission-Prediction)  
  *Description:* Predicting 30-day inpatient readmission across 101,766 encounters. Tackling 97% structural missingness in weight, multi-class imbalance, and high-cardinality ICD-9 codes.

---

## 8. The Minimalist Data Manifesto (Topic 01 Edition)

My biggest takeaway from Topic 01 isn't a complex equation. It is a philosophy of humility and observation.

Just like *Minimalist Baker* transformed cooking by stripping away fussy 30-step recipes and focusing on simple, honest, foolproof ingredients — data science is vastly more rewarding when you strip away unexamined assumptions and learn to look:

1. **Look before you calculate:** As Francis Anscombe proved in 1973, numbers without graphs can actively deceive you.
2. **Protect with Resistance:** Trust the median to reflect the bulk of reality; don't let single outliers hijack your story.
3. **Respect the Residuals:** $	ext{Data} = 	ext{Fit} + 	ext{Residual}$. What your model failed to capture is often where the genuine breakthrough hides.
4. **Re-express fearlessly:** Raw scales are not sacred. Transformations like logarithms aren't cheating — they are the magnifying glass that lets patterns emerge.
5. **Never abandon a dataset out of panic:** Missing values and skewness are not bugs. They are the data speaking to you. Stop hopping, start exploring.

---

## 9. Curated Inspiration & Recommended Reading
*(Required by DS311 PIT Assignment Rubric)*

* [33 Blog Examples to Inspire Your Next Blog Post](https://nealschaffer.com/blog-post-examples/) — Neal Schaffer
* [11 Best Free Blog Sites in 2026: Build and Grow Your Blog](https://www.ryrob.com/free-blogging-sites/) — Ryan Robinson
* [Minimalist Baker: Simple Recipes & Uncluttered Design](https://minimalistbaker.com/) — Inspiration for the clean, earth-toned aesthetic
