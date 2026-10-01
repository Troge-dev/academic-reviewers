# DS314: GENERATIVE AI & LARGE LANGUAGE MODEL SYSTEMS
## Comprehensive Academic Reviewer & Laboratory Companion Guide
### Week 4: Prompt Engineering Practice (Anatomy, In-Context Learning, Persona Steering & Decoding Thermodynamics)

---

### EXECUTIVE COMPANION PREFACE & METHODOLOGY

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       LABORATORY TOPIC ROADMAP & ANALYTICAL LENSES                     │
│               Course: DS314 — Generative AI (Google AI Studio / LangChain Edition)     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • FOCUS 1: STRUCTURAL & EMPIRICAL ANALYSIS                                             │
│   - Activity 9.1: Length & Format Constraints (Word limits, structural compliance)     │
│   - Activity 9.2: Test Set Construction & Subgroup Difficulty Analysis                 │
│   - Activity 9.3: Hostile Cadence Analysis (Substantive fact preservation vs. tone)   │
│   - Activity 9.4: Creative Generation Divergence (Temperature 0.0 vs. High Temp)       │
│   - Activity 9.5: Empirical Reflection on Output Sensitivity & Reliability Bounds      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • FOCUS 2: SEMANTIC & ECONOMIC ANALYSIS                                                │
│   - Activity 9.1: Tone & Vocabulary Constraints (Banned lexicon, required phrasing)    │
│   - Activity 9.2: Error Taxonomy (Content vs. Format Failure) & Token Economics / ROI  │
│   - Activity 9.3: Empathetic Cadence Analysis (Polite padding vs. informational advice)│
│   - Activity 9.4: Deterministic Classification Resilience under High Temperature       │
│   - Activity 9.5: Empirical Reflection on Output Sensitivity & Reliability Bounds      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

This companion document serves as the academic, theoretical, and pedagogical reviewer for the **DS314 Week 4 Laboratory Task: Prompt Engineering Practice**.

In strict accordance with pedagogical ethics, **this document does not write the solutions or pre-fill the notebook cells**. Instead, it provides the uncompromised theoretical foundations, architectural mechanisms, diagnostic decision trees, and mathematical explanations necessary for the duo to master the laboratory experiments and achieve full marks across the 100-point rubric.

For every section of the laboratory, a four-tier analytical framework is established:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               FOUR-TIER REVIEWER FRAMEWORK                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. ACADEMIC CONTEXT & THEORETICAL DEEP DIVE                                           │
│    Transformer conditioning, autoregressive attention, Bayesian in-context learning,   │
│    logits, softmax temperature thermodynamics, and tokenomics.                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. KEY DEFINITION GLOSSARY                                                            │
│    Rigorous definitions of every technical term, mechanism, and design parameter.      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. APPLIED ENGINEERING SCENARIOS & FAILURE MODES                                       │
│    Real-world software engineering bugs, production drift, jailbreak dynamics, and     │
│    token-cost trade-offs in industry deployments.                                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. DIAGNOSTIC REVIEWER QUESTIONS & DUAL-LENS AUDITS                             │
│    Critical-thinking Q&As, evaluation templates, and rubric-aligned reflection guides  │
│    engineered for structural and semantic analysis.                                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           WEEK 4 LABORATORY ROADMAP & SCORING                          │
├─────────┬──────────────────────────────────────────┬────────┬──────────────────────────┤
│ SECTION │ CORE EXPERIMENT                          │ POINTS │ KEY CHECKPOINTS          │
├─────────┼──────────────────────────────────────────┼────────┼──────────────────────────┤
│ Part 1  │ Prompt Duel: The Anatomy Ladder          │ 25 pts │ V1→V2→V3 Evolution,      │
│         │ (Raw Task → Structured → Hard Limits)    │        │ Length & Tone Checks     │
├─────────┼──────────────────────────────────────────┼────────┼──────────────────────────┤
│ Part 2  │ Zero-Shot vs. Few-Shot Classifier        │ 30 pts │ 8 Test Cases, No Leak,   │
│         │ (Exemplar Conditioning & Token Costs)    │        │ Content vs Format Error  │
├─────────┼──────────────────────────────────────────┼────────┼──────────────────────────┤
│ Part 3  │ System Instruction Duel: Angry vs. Nice  │ 15 pts │ Same Invariant Facts,    │
│         │ (Persona Steering & Brand Vulnerability) │        │ Tone vs Role Separation  │
├─────────┼──────────────────────────────────────────┼────────┼──────────────────────────┤
│ Part 4  │ Temperature Duel: Low vs. High           │ 15 pts │ T=0.0 vs T=1.8, Creative │
│         │ (Sampling Thermodynamics & Logits)       │        │ vs Deterministic Duality │
├─────────┼──────────────────────────────────────────┼────────┼──────────────────────────┤
│ Part 5  │ Individual Duo Reflections               │ 15 pts │ 80–120 Words each,       │
│         │ (Empirical Metrics & Prompt Trust Bounds)│        │ Grounded in Real Data    │
├─────────┴──────────────────────────────────────────┼────────┼──────────────────────────┤
│ TOTAL                                              │ 100 pts│ Verified Run Outputs     │
└────────────────────────────────────────────────────┴────────┴──────────────────────────┘
```

---

## MODULE 01: THE ANATOMY OF A PROMPT & ITERATIVE REFINEMENT
### Laboratory Section: `9.1 Part 1: Prompt Duel — Anatomy Ladder`
### Focus: *Deconstructing Context, Role, and Algorithmic Constraints*

```
Ladder Progression Architecture:
[V1: Raw Task Only]  ──▶  [V2: Task + Role + Context + Target Format]  ──▶  [V3: V2 + Checkable Hard Constraints]
       │                                     │                                            │
  High Entropy                          Contextualized                             Programmatically
  Broad Variance                       Audience-Aligned                               Verifiable
```

---

#### 1. Academic Context & Theoretical Deep Dive

##### The Autoregressive Conditioning Mechanism
Large Language Models (LLMs) such as Gemini 3.1 Flash Lite are autoregressive probabilistic sequence predictors. Formally, given an input sequence of tokens $X = (x_1, x_2, \dots, x_n)$, the model computes the probability distribution of the next token $x_{n+1}$:

$$P(x_{n+1} \mid x_1, x_2, \dots, x_n) = \text{Softmax}\left(\frac{W_u h_n}{\tau}\right)$$

Where $h_n$ represents the final contextualized hidden state produced by the multi-head self-attention layers, $W_u$ is the language model unembedding matrix, and $\tau$ is the sampling temperature.

When a prompt is supplied, it acts as an **information-theoretic conditioning prior**. A raw, under-specified prompt (Version 1) provides a broad, unconstrained conditioning space. Consequently, the model defaults to the statistical average of its pretraining corpus, yielding responses that are typically generic, verbose, and stylistically neutral.

##### The Five Functional Pillars of Prompt Anatomy
Systematic prompt engineering structures the input context into five discrete functional components:

1. **Role / Identity:** Conditions the model's persona, expertise level, and communicative stance (e.g., *"You are a senior academic advisor..."*). Statistically, setting a role elevates the probability of domain-specific terminology and appropriate syntactic complexity.
2. **Task / Primary Directive:** The core operational verb and objective (e.g., *"Draft an email requesting a deadline extension"*).
3. **Context / Operational Background:** The situational parameters, external environment, and rationale (e.g., *"The student was hospitalized with acute bronchitis during the midterm week..."*). Context drastically narrows the candidate output space.
4. **Target Format:** The explicit structural output schema (e.g., *"Subject line, formal salutation, 2 body paragraphs, bulleted timeline, sign-off"*).
5. **Hard Constraints (Negative & Positive):** Inviolable boundaries governing length, vocabulary, structure, and style (e.g., *"Maximum 120 words. Do not use the word 'sorry'. Include the exact phrase 'medical documentation attached'"*).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE PROMPT ANATOMY LADDER ARCHITECTURE                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ LEVEL 1: TASK (V1)                                                                     │
│ • Single unadorned command.                                                            │
│ • Leaves persona, audience, tone, length, and format to random prior defaults.         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ LEVEL 2: ROLE + CONTEXT + FORMAT (V2)                                                  │
│ • Establishes who is speaking, to whom, under what circumstances, and in what layout.  │
│ • Output shifts from generic boilerplate to professional, context-aware discourse.    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ LEVEL 3: CHECKABLE CONSTRAINTS (V3)                                                    │
│ • Injects deterministic boundaries that can be verified algorithmically.              │
│ • Pillar A: Length / Structural Metric (word count cap, sentence count, bullet count). │
│ • Pillar B: Tone / Lexical Rule (banned vocabulary words, mandatory anchor phrases).   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

##### The Pathology of Negative Constraints and Word Count Drift
A fundamental insight in prompt engineering is that **LLMs process text in tokens, not words or characters**. Tokens represent sub-word statistical fragments (e.g., an average of ~0.75 words per token in English).

Because LLMs generate tokens autoregressively without a backward revision loop (unless external reasoning or tree-search decoding is utilized), they cannot maintain a global forward counter of total generated words. Consequently:
* **Word Count Imprecision:** Prompts specifying *"Write exactly 100 words"* routinely fail, producing between 75 and 130 words. Models navigate length via learned soft correlations rather than explicit counting. To establish a checkable length constraint in V3, an **asymmetric upper ceiling** (e.g., *"under 120 words"*) or a **discrete structural count** (e.g., *"exactly 3 bullet points"*) is dramatically more reliable than an exact word quota.
* **Negative Constraint Failure (The "Pink Elephant" Paradox):** Directives such as *"Do not use the word 'urgent'"* require the self-attention mechanism to attend to the banned token. By mentioning the token in the prompt, its semantic vector is activated in the attention context, paradoxically increasing the probability of its lexical neighborhood. Effective negative conditioning requires explicit, punitive negative framing combined with mandatory replacement vocabulary.

---

#### 2. Key Definition Glossary

* **Prompt Anatomy:** The systematic decomposition of a generative prompt into functional constituents: Role, Task, Context, Format, and Constraints.
* **Autoregressive Generation:** A process wherein a neural network generates text token-by-token, feeding each newly predicted token back into its input context for subsequent predictions.
* **In-Context Conditioning:** The mathematical restriction of the language model's output probability distribution through the tokens present within its input context window.
* **Checkable Constraint:** An objective, falsifiable constraint whose satisfaction can be verified via a deterministic algorithm (e.g., boolean substring search, word-count splitting) rather than subjective human impression.
* **Tokenization:** The process of converting raw natural language text into discrete numerical identifiers (tokens) based on a vocabulary dictionary (e.g., Byte-Pair Encoding or SentencePiece).
* **Negative Lexical Constraint:** An explicit instruction prohibiting the appearance of specific words, phrases, or linguistic markers within the generated text.

---

#### 3. Applied Engineering Scenarios & Failure Modes

* **Production Support Bot Failure:** A company instructs its customer service LLM: *"Help the customer reset their password. Be brief. Do not say sorry."* The model responds with 450 words of corporate fluff and apologizes twice (*"We deeply apologize for the inconvenience..."*). Why? The prompt used soft adjectives (*"brief"*) rather than a checkable metric, and the negative constraint was overpowered by the dominant polite conversational prior learned during RLHF fine-tuning.
* **The V1 $\to$ V2 $\to$ V3 Industrial Evolution:**
  * *V1:* `"Write an incident report for a database outage."` (Result: 6-page generic fictional essay).
  * *V2:* `"You are a Site Reliability Engineer. Write an internal post-mortem incident report for a 15-minute PostgreSQL outage caused by connection pool exhaustion for the CTO."` (Result: Realistic structure, correct tone, but runs 800 words with excessive details).
  * *V3:* `"[V2] + Constraints: Must not exceed 150 words. Must contain sections: ROOT CAUSE, IMPACT, ACTION ITEMS. Banned words: 'unfortunate', 'accidentally', 'sorry'."` (Result: Concise, audit-ready, deterministic).

---

#### 4. Diagnostic Reviewer Questions & Partner Lens Exercises

##### Diagnostic Question 1: Algorithmic Verification
*Why does the laboratory utilize a programmatic `check()` function instead of visual inspection alone?*
> **Answer:** Human subjective evaluation suffers from cognitive bias and visual scanning errors. A programmatic check (`len(text.split())` and `b.lower() in text.lower()`) enforces **deterministic verification**. It operationalizes prompt engineering as an empirical science: a prompt is not "good" because it looks pleasant; it is valid because it compiles against specifications.

##### Structural Lens Audit (Length & Format Compliance)
* **Analytical Focus:** Inspect the word count output of V2 versus V3.
* **Diagnostic Audit:**
  1. Did V3 successfully compress or reshape the output to stay within the ceiling specified in `check()`?
  2. If the prompt demanded a 3-bullet list or a maximum of 100 words, did the model truncate essential factual context, or did it increase informational density?
  3. *Key Takeaway for Reflection:* When length constraints are imposed, the model is forced to eliminate conversational pleasantries and increase lexical density per token.

##### Semantic Lens Audit (Tone & Vocabulary Compliance)
* **Analytical Focus:** Inspect the banned and required strings in V2 versus V3.
* **Diagnostic Audit:**
  1. Did V3 completely eliminate the banned word(s)? Did the model substitute them with awkward synonyms, or did it restructure its syntax entirely?
  2. If a required anchor phrase was mandated, where did the model position it (introduction, body, or conclusion)?
  3. *Key Takeaway for Reflection:* Banned words test the model's ability to override its conversational priors (such as habitual apologies or common transition words) without degrading grammatical coherence.

---

#### 5. Comparative Synthesis Matrix: The Anatomy Ladder

```
┌──────────┬─────────────────────────────┬───────────────────────────┬────────────────────────────────┐
│ VERSION  │ COMPONENTS INCLUDED         │ OUTPUT CHARACTERISTICS    │ VERIFICATION MECHANISM         │
├──────────┼─────────────────────────────┼───────────────────────────┼────────────────────────────────┤
│ V1: Task │ Task only                   │ Verbose, generic, neutral,│ None (Subjective impression)   │
│          │ (e.g., "Write an email...") │ unbounded length.         │                                │
├──────────┼─────────────────────────────┼───────────────────────────┼────────────────────────────────┤
│ V2: Rich │ Task + Role + Context +     │ Professional, stylized,   │ Qualitative alignment check    │
│ Context  │ Format                      │ realistic domain schema.  │ (Looks like an authentic email)│
├──────────┼─────────────────────────────┼───────────────────────────┼────────────────────────────────┤
│ V3: Hard │ V2 + Checkable Word Cap +   │ Dense, concise, compliant,│ Programmatic `check()` script  │
│ Limits   │ Banned / Required Lexicon   │ zero banned vocabulary.   │ (Falsifiable boolean tests)    │
└──────────┴─────────────────────────────┴───────────────────────────┴────────────────────────────────┘
```

---

## MODULE 02: IN-CONTEXT LEARNING: ZERO-SHOT VS. FEW-SHOT CLASSIFICATION
### Laboratory Section: `9.2 Part 2: Zero-Shot vs. Few-Shot Classifier`
### Focus: *Exemplar Conditioning, Label Calibration, and Token Economics*

```
Classification Pipeline Architecture:
                            ┌────────────────────────────────────────┐
                            │ LAB_INSTRUCTIONS (System Prompt)       │
                            │ Role + Label Definitions + Format Rule │
                            └──────────────────┬─────────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
  ┌─────────────────────────┐                                    ┌───────────────────────────┐
  │ ZERO-SHOT PROMPT        │                                    │ FEW-SHOT PROMPT           │
  │ System + Test Message   │                                    │ System + 3-5 Exemplars    │
  │ Cost: ~40-60 tokens     │                                    │        + Test Message     │
  └────────────┬────────────┘                                    │ Cost: ~180-300 tokens     │
               │                                                 └─────────────┬─────────────┘
               ▼                                                               ▼
  ┌─────────────────────────┐                                    ┌───────────────────────────┐
  │ Output: Predicted Label │                                    │ Output: Predicted Label   │
  └─────────────────────────┘                                    └───────────────────────────┘
```

---

#### 1. Academic Context & Theoretical Deep Dive

##### The Mechanics of In-Context Learning (ICL)
Pioneered formally by Brown et al. (2020) with GPT-3, In-Context Learning (ICL) is the emergent capability of autoregressive language models to execute tasks when prompted with input-output demonstration pairs (*exemplars*) without updating the model's underlying neural weights ($\Delta W = 0$).

From a Bayesian perspective, ICL can be understood as **implicit Bayesian inference**. The pre-trained LLM possesses a vast mixture of latent concept models acquired during self-supervised pretraining. The prompt $P$—specifically the system instructions and few-shot exemplars $(x_1, y_1), (x_2, y_2), \dots, (x_k, y_k)$—serves as evidence that isolates a specific latent task $T$:

$$P(y \mid x, \text{Prompt}) = \int P(y \mid x, T) P(T \mid \text{Prompt}) \, dT$$

##### Zero-Shot vs. Few-Shot Conditioning
1. **Zero-Shot Classification:** Relies purely on the model's semantic understanding of the label names and the descriptive rules in `LAB_INSTRUCTIONS`. If a label name is polysemous or if the decision boundary between categories is nuanced, zero-shot prompts frequently suffer from **label bias** or **priors skew** (e.g., favoring the more common word in internet text regardless of context).
2. **Few-Shot Classification:** Exemplars act as **distributional anchors**. They achieve three simultaneous functions:
   - **Label Space Calibration:** Demonstrates the exact valid set of target strings, eliminating extraneous conversational filler (e.g., outputting just `"urgent"` instead of `"The category of this message is urgent."`).
   - **Decision Boundary Disambiguation:** Resolves borderline ambiguities that text instructions struggle to describe.
   - **Stylistic & Formatting Enforcement:** Forces the autoregressive generation into an identical structural pattern matching the demonstration suffix.

##### Test Set Purity and The Data Leakage Catastrophe
In machine learning, **data leakage** occurs when information from the evaluation/test dataset contaminates the training or conditioning context.
In prompt engineering:
* If a message from `lab_test_set` is duplicated inside `lab_examples`, the model is not generalizing; it is performing trivial in-context memorization.
* The laboratory notebook enforces this via an automated programmatic assertion:
  ```python
  rendered_few_shot = lab_few_shot.invoke({"message": "PLACEHOLDER"}).to_string()
  assert not any(t["message"] in rendered_few_shot for t in lab_test_set), \
      "A test message leaked into your few-shot examples — pick different examples."
  ```
* **Exemplar Design Standard:** Exemplars must be *synthetically distinct* but *distributionally representative* of the real edge cases.

##### The Dichotomy of Errors: Content Errors vs. Format Errors
When auditing model predictions against expected ground-truth labels, errors must be categorized into two fundamentally distinct classes:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              ERROR TAXONOMY FOR LLM EVALUATION                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. CONTENT ERROR (Semantic Misclassification)                                          │
│ • The model outputs a clean, valid label from the allowed list, but it picked the WRONG│
│   category (e.g., expected 'billing', predicted 'technical_support').                  │
│ • Cause: Genuine semantic ambiguity, complex edge cases, or weak contextual reasoning.│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. FORMAT ERROR (Syntactic / Structural Degradation)                                   │
│ • The model understood the concept, but failed to adhere to the output constraint      │
│   (e.g., expected 'urgent', got 'Label: urgent', 'URGENT.', or a complete sentence).   │
│ • Cause: Prompt failed to suppress conversational tendencies; lack of few-shot anchors.│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

##### Token Economics and The Cost-Accuracy Pareto Frontier
Every API call incurs financial and computational costs proportional to prompt tokens ($T_{in}$) and generated completion tokens ($T_{out}$).

$$\text{Total Cost} = (N \times T_{in} \times \text{Cost}_{in}) + (N \times T_{out} \times \text{Cost}_{out})$$

Where $N$ is the number of processed requests.
* **The Few-Shot Tax:** Adding 4 exemplars increases prompt token count from ~50 tokens to ~250 tokens per request—a **400% to 500% increase in input cost and memory consumption**.
* **The Engineering ROI Question:** If zero-shot achieves 87.5% (7/8) accuracy and few-shot achieves 87.5% (7/8) or 100% (8/8), did the marginal 12.5% accuracy gain justify multiplying production token costs by $4\times$ or $5\times$? In high-throughput industrial pipelines (processing 10 million requests/day), few-shot costs millions of dollars; in low-volume mission-critical pipelines, few-shot is mandatory.

---

#### 2. Key Definition Glossary

* **Zero-Shot Prompting:** Presenting the model with a task description and an input instance without any completed demonstrations.
* **Few-Shot Prompting (Exemplar Conditioning):** Providing the model with a small number of input-output demonstrations within the prompt context to guide its execution.
* **Data Leakage (Prompt Contamination):** An experimental flaw wherein test examples designated for performance measurement are inadvertently included in the prompt's few-shot exemplars.
* **Content Error:** A failure mode where the generated label is syntactically valid but semantically incorrect according to the ground-truth annotation.
* **Format Error:** A failure mode where the generated text violates structural constraints (case sensitivity, extraneous tokens, conversational filler), even if the underlying semantic intent was correct.
* **Prompt Token Overhead:** The baseline token count consumed by system instructions and demonstration exemplars before the actual runtime user query is ingested.
* **LangChain Expression Language (LCEL):** A declarative orchestration syntax (`prompt | model | parser`) that composes modular components into production runtime chains.

---

#### 3. Applied Engineering Scenarios & Industrial Edge Cases

* **Financial Transaction Classifier Failure:** A fintech platform builds a triage bot with categories: `[Fraud, Inquiry, Dispute]`.
  * *Zero-Shot Failure:* A customer writes: *"I see a charge from Netflix that I didn't authorize, please cancel it."* The zero-shot model outputs: `"Dispute (possibly Fraud)"`. Because the string contains extra words, the downstream regex fails, crashing the transaction router.
  * *Few-Shot Resolution:* By supplying 3 clean exemplars demonstrating `Message -> Label`, the model outputs strictly `"Fraud"`. The format error is eliminated.
* **The "Author Hardness" Disparity in Engineering Teams:** When two engineers create test sets independently, one engineer frequently writes subtle, sarcastic, or multi-intent messages (e.g., *"Thanks for charging me twice, wonderful software!"*), while the other writes straightforward statements. This exposes the boundary between clear intent and human linguistic ambiguity.

---

#### 4. Diagnostic Reviewer Questions & Partner Lens Exercises

##### Diagnostic Question 1: Few-Shot Token Justification
*Under what empirical conditions is few-shot prompting economically justified over zero-shot prompting?*
> **Answer:** Few-shot prompting is justified when: (1) Zero-shot produces format errors that break automated downstream code; (2) The task involves proprietary, non-standard taxonomies that the model's pretraining corpus cannot infer from label names alone; and (3) The cost of a classification error exceeds the $3\times - 5\times$ token overhead cost.

##### Structural Lens Audit (Author Difficulty Breakdown)
* **Analytical Focus:** Analyze `accuracy_by_author(rows)`.
* **Diagnostic Audit:**
  1. Did the model score higher on Author A's 4 messages or Author B's 4 messages?
  2. If Partner A wrote messages that caused model errors, were those messages inherently multi-intent (e.g., a message that is both a complaint and a feature request)?
  3. *Key Takeaway for Reflection:* Disparities in author accuracy reveal that label definitions in `LAB_INSTRUCTIONS` were underspecified for complex edge cases.

##### Semantic Lens Audit (Error Taxonomy & Token Economics)
* **Analytical Focus:** Inspect the `show(rows)` output and token counters.
* **Diagnostic Audit:**
  1. Examine every failed row: Did the model output a wrong word (content error) or the right word with markdown/extra tokens (format error)?
  2. Compare `prompt_tokens(lab_zero_shot)` versus `prompt_tokens(lab_few_shot)`.
  3. *Key Takeaway for Reflection:* If both zero-shot and few-shot achieve identical accuracy (e.g., 8/8), few-shot failed to provide economic return on investment (ROI) because it expended 400% more tokens for zero marginal gain.

---

#### 5. Comparative Evaluation Matrix: Zero-Shot vs. Few-Shot

```
┌─────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ EVALUATION DIMENSION    │ ZERO-SHOT CLASSIFICATION      │ FEW-SHOT CLASSIFICATION       │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Context Window Footprint│ Minimal (~30 – 80 tokens)     │ Moderate (~150 – 400 tokens)  │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Marginal Cost per Call  │ Low                           │ 3× to 6× higher               │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Formatting Reliability  │ Vulnerable to conversational  │ Highly deterministic;         │
│                         │ boilerplate and explanations  │ mimics exemplar syntax        │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Ambiguity Disambiguation│ Relies strictly on semantic   │ Disambiguates nuances through │
│                         │ clarity of label names        │ concrete input-output anchors │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Risk of Data Leakage    │ Non-existent                  │ Present if exemplars overlap  │
│                         │                               │ with the evaluation test set  │
└─────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## MODULE 03: SYSTEM INSTRUCTIONS, PERSONA STEERING & SAFETY INVARIANTS
### Laboratory Section: `9.3 Part 3: System Instruction Duel — Angry vs. Nice`
### Focus: *The Dichotomy Between Behavioral Persona and Invariant Substantive Advice*

```
Tone Modulation Architecture:
                                  ┌───────────────────────────┐
                                  │ SHARED CUSTOMER SITUATION │
                                  │ (Facts, Invariant Steps)  │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
  ┌──────────────────────────────┐                              ┌──────────────────────────────┐
  │ SYSTEM: ANGRY / IMPATIENT    │                              │ SYSTEM: NICE / APOLOGETIC    │
  │ Harsh tone, terse, irritated │                              │ Warm tone, polite, reassuring│
  │ MANDATE: Preserve Real Steps │                              │ MANDATE: Preserve Real Steps │
  └──────────────┬───────────────┘                              └──────────────┬───────────────┘
                 │                                                             │
                 ▼                                                             ▼
  ┌──────────────────────────────┐                              ┌──────────────────────────────┐
  │ ANGRY REPLY                  │                              │ NICE REPLY                   │
  │ Aggressive phrasing, but     │                              │ Gentle phrasing, containing  │
  │ identical factual advice.    │                              │ identical factual advice.    │
  └──────────────────────────────┘                              └──────────────────────────────┘
```

---

#### 1. Academic Context & Theoretical Deep Dive

##### The Dual-Role Prompting Paradigm: System vs. Human Messages
In modern chat-tuned LLM architectures, input contexts are segmented into distinct conversational roles:
* `SystemMessage` (System Instruction): Establishes the global operating envelope, immutable behavioral constraints, persona, and meta-rules of the model. In the model's internal attention mask, system tokens are positioned to exert pervasive influence over the entire sequence.
* `HumanMessage` (User Turn): Represents the immediate, dynamic runtime query or input data.

##### The Engineering Principle of Semantic Invariance
A critical challenge in agentic AI is **persona steerability versus informational integrity**.
When modifying an agent's persona (e.g., from professional to angry, sarcastic, or apologetic):
* **The Surface Layer (Style / Persona):** Lexical choice, sentence length, punctuation, emotional coloring, and conversational cadence.
* **The Core Invariant Layer (Substance / Facts):** The ground-truth instructions, procedural troubleshooting steps, policy mandates, and safety warnings.

Formally, let an advice function be $f(M, S)$, where $M$ is the customer message and $S$ is the system instruction. If $I(A)$ represents the informational content of response $A$, true semantic invariance requires:

$$I(f(M, S_{angry})) \equiv I(f(M, S_{nice}))$$

##### Persona Drift: When Tone Eats Substance
In poorly engineered system prompts, **tone cannibalizes substance**. When instructed to be intensely angry or impatient, the model's attention weights can become overwhelmingly saturated with emotionally combative tokens (e.g., insults, exasperated exclamations, rhetorical questions). Consequently, the model drops essential troubleshooting steps, truncates URLs, or omits safety caveats.
Conversely, an excessively polite system instruction often pads the output with repetitive apologies, obscuring the actionable advice beneath cognitive clutter.

##### Corporate Brand Safety and "The Screenshot Vulnerability"
From an enterprise software engineering standpoint, exposing an unregulated angry or aggressive persona to end-users creates severe reputational risk:
* **The Screenshot Vulnerability:** Any text generated by an enterprise model can be screenshotted and disseminated across social or news media. Even if the factual advice was technically accurate, an adversarial or hostile tone constitutes a brand disaster and potential liability.
* **Tone as a Constraint vs. Tone as a Role:**
  * If tone is treated purely as a *role* (`"You are an angry support agent"`), the model has broad license to drift.
  * If tone is governed by *constraints* (`"Express extreme urgency and impatience, BUT you MUST explicitly state steps A, B, and C and never use profanity"`), the informational payload remains protected.

---

#### 2. Key Definition Glossary

* **System Instruction (Meta-Prompt):** A privileged prompt layer that sets the behavioral rules, constraints, persona, and boundaries for a language model session.
* **Semantic Invariance:** The preservation of identical factual information, procedural steps, and logical meaning across text variations differing in style, tone, or phrasing.
* **Persona Steering:** Guiding an LLM to adopt a specific communicative personality, emotional disposition, or professional identity.
* **Substance Cannibalization (Tone Drift):** A failure mode where the stylistic or emotional demands of a persona cause the model to omit, corrupt, or truncate the core factual advice.
* **Brand Vulnerability (Screenshot Risk):** The operational danger that an AI model generates inflammatory, hostile, or inappropriate responses that can be captured and published, causing institutional harm.

---

#### 3. Applied Engineering Scenarios & Enterprise Realities

* **Automated IT Helpdesk Scenario:** An IT support agent responds to: *"My VPN keeps disconnecting every 5 minutes."*
  * *Substantive Invariant:* (1) Restart router; (2) Re-authenticate via 2FA app; (3) Verify certificate expiry date.
  * *Angry Prompt Failure:* The model writes: *"Stop bothering IT with basic issues! If you can't keep your internet running, talk to your ISP. We have real servers to maintain!"* $\to$ **Catastrophic Failure:** Zero actionable steps provided. Tone completely consumed substance.
  * *Calibrated Angry Implementation:* *"Seriously? Read carefully because I won't repeat this: First, reboot your router right now. Second, reopen your 2FA app and re-authenticate. Third, check if your security cert is expired. Do those three things before opening another ticket."* $\to$ **Valid:** Harsh tone, but all 3 invariant facts preserved.

---

#### 4. Diagnostic Reviewer Questions & Partner Lens Exercises

##### Diagnostic Question 1: Tone Mechanism
*Is tone primarily a function of 'role', 'constraints', or both?*
> **Answer:** Tone is initiated through **role** (establishing the behavioral archetype), but it must be governed by **constraints** (setting boundary conditions). Without constraints, role-based tone inevitably causes hallucination or substantive omissions. Therefore, production-grade persona steering requires both.

##### Structural Lens Audit (Substantive Preservation in Angry Persona)
* **Analytical Focus:** Scrutinize the output of `system_angry`.
* **Diagnostic Audit:**
  1. Did the angry reply contain every single factual step present in the prompt's underlying scenario?
  2. Did the model add hostile fabrications or refuse to assist?
  3. *Key Takeaway for Reflection:* A successful angry prompt proves that emotional coloring and logical execution can coexist if the prompt's instructional hierarchy is strictly enforced.

##### Semantic Lens Audit (Informational Density in Nice Persona)
* **Analytical Focus:** Scrutinize the output of `system_nice`.
* **Diagnostic Audit:**
  1. Strip away all polite pleasantries (*"I am so terribly sorry", "Thank you so much for your patience"*). What remains?
  2. Is the substantive advice identical to the angry version, or did the polite persona soften negative facts (e.g., downplaying a delay or refund denial)?
  3. *Key Takeaway for Reflection:* Polite personas often dilute clarity by wrapping actionable instructions in excessive conversational padding.

---

#### 5. Architectural Comparison Matrix: Persona Steering vs. Fact Retention

```
┌─────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ OPERATIONAL AXIS        │ ANGRY / IMPATIENT PERSONA     │ NICE / APOLOGETIC PERSONA     │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Primary Failure Mode    │ Substance omission; abusive   │ Verbose padding; softening    │
│                         │ or hostile refusal to help    │ of critical policy truths     │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Enterprise Viability    │ Zero (Extreme screenshot risk │ High for customer relations,  │
│                         │ and brand liability)          │ but risks cognitive fatigue   │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Lexical Efficiency      │ High (Terse, clipped, direct) │ Low (High token consumption   │
│                         │                               │ spent on empathetic phrases)  │
├─────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Invariant Check         │ Must provide identical next   │ Must provide identical next   │
│                         │ steps despite hostile framing │ steps despite apologetic fluff│
└─────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## MODULE 04: STOCHASTIC SAMPLING DYNAMICS & TEMPERATURE THERMODYNAMICS
### Laboratory Section: `9.4 Part 4: Temperature Duel — Predictable vs. Creative`
### Focus: *Softmax Logit Scaling, Greedy Decoding, and Task Determinism*

```
Softmax Temperature Thermodynamics:
                z_i (Raw Logits from Transformer)
                              │
               Scale by 1/T:  z_i / T
                              │
                    Softmax Distribution:
               P(w_i) = exp(z_i/T) / ∑ exp(z_j/T)
                              │
       ┌──────────────────────┴──────────────────────┐
       ▼                                             ▼
  Temperature T = 0.0                           Temperature T = 1.8
  (Argmax / Greedy Selection)                   (Flattened Softmax Distribution)
       │                                             │
  Peak Probability = 1.0                        High Entropy Sampling
  Zero Entropy                                  Rare Tokens Activated
  100% Deterministic Repetition                 High Creativity / Hallucination Risk
```

---

#### 1. Academic Context & Theoretical Deep Dive

##### The Softmax Temperature Function
At the conclusion of the Transformer forward pass, the model outputs an unnormalized vector of real numbers called **logits** $\mathbf{z} = (z_1, z_2, \dots, z_{|V|})$, where $|V|$ is the size of the vocabulary.

To convert logits into a valid probability distribution, the **Softmax function with Temperature scaling** ($T > 0$) is evaluated:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$

Where $T$ represents the temperature parameter:
* **As $T \to 0$ (Cold / Greedy Limit):** The transformation approaches an $\text{argmax}$ indicator function. The token with the highest logit receives probability $P(w_{\max}) \to 1.0$, while all other tokens collapse to $0.0$.
  $$\lim_{T \to 0^+} P(w_i) = \begin{cases} 1 & \text{if } i = \arg\max_j z_j \\ 0 & \text{otherwise} \end{cases}$$
  Generation becomes completely deterministic. Running the exact same prompt 5 times at $T=0.0$ will yield the exact same completion every single run.
* **At $T = 1.0$:** The distribution reflects the model's raw unscaled training probabilities.
* **As $T \to \infty$ (Hot / High Entropy Limit):** All logits are scaled toward zero ($z_i / T \to 0$). Because $\exp(0) = 1$, the distribution flattens into a **uniform distribution**:
  $$\lim_{T \to \infty} P(w_i) = \frac{1}{|V|}$$
  Every token in the vocabulary becomes equally likely, resulting in complete syntactic incoherence and hallucination.
* **At High Temperatures ($T \in [1.5, 2.0]$):** The probability mass of the top candidates is reduced, elevating the selection probability of lower-ranked tokens (the "tail" of the distribution). This produces rich lexical variety, unique metaphoric connections, and stylistic diversity in creative tasks—but introduces severe instability in structured tasks.

##### The Task Duality: Creative Synthesis vs. Single-Label Classification
The core experimental objective of Part 4 is testing the impact of high temperature across two fundamentally distinct operational regimes:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               THE OPERATIONAL TASK DUALITY                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TASK 1: OPEN-ENDED CREATIVE GENERATION (High Intrinsic Entropy)                        │
│ • Characteristics: Many equally valid outputs; broad solution space.                   │
│ • Behavior at T=0.0: Repetitive, clichéd, produces the identical output 5 times.       │
│ • Behavior at T=1.8: Highly diverse, surprising metaphors, distinct name variations.   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TASK 2: DETERMINISTIC CLASSIFICATION (Zero Intrinsic Entropy)                          │
│ • Characteristics: Exactly ONE correct ground-truth label (e.g., 'urgent').            │
│ • Behavior at T=0.0: Reliable, deterministic output matching the dominant logit.       │
│ • Behavior at T=1.8: If the margin of separation between the top logit and runner-up   │
│   logit is narrow, the label will FLIP randomly between runs, causing failure.         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

##### The Margin of Separation in Logit Space
Why does a classifier sometimes **not flip** even at $T=1.8$?
* If a customer message is unambiguously urgent (e.g., *"Help, my server is on fire and losing data!"*), the model's attention heads assign an overwhelming logit advantage to `"urgent"` (e.g., $z_{\text{urgent}} = 18.5$, while $z_{\text{inquiry}} = 2.1$).
* Even after dividing by $T=1.8$, the logit gap $\Delta z / T = (18.5 - 2.1) / 1.8 = 9.11$ remains massive. In the exponential space, $\exp(9.11) \approx 9000$, meaning `"urgent"` still retains over 99% of the probability mass!
* **The Diagnostic Rule:** High temperature causes label flipping **only when the input message is ambiguous**, meaning the top two competing labels had close initial logits ($\Delta z \approx 0$).

---

#### 2. Key Definition Glossary

* **Logits:** The raw, unnormalized vector of real-valued predictions generated by the final linear layer of a Transformer before probability normalization.
* **Softmax Function:** A mathematical normalized exponential function that maps an arbitrary vector of real numbers into a probability distribution summing to 1.
* **Decoding Temperature ($T$):** A hyperparameter that scales logits prior to the softmax calculation, modulating the entropy and randomness of token selection.
* **Greedy Decoding:** A sampling strategy that deterministically selects the single token with the highest predicted probability at each step ($T=0.0$).
* **Sampling Entropy:** A measure of the uncertainty or unpredictability in the model's token probability distribution.
* **Margin of Separation ($\Delta z$):** The numerical distance between the logit of the most probable token and the logit of the second-most probable token.

---

#### 3. Applied Engineering Scenarios & Industrial Edge Cases

* **Financial Information Extraction Disaster:** An enterprise deploys an LLM to extract numerical revenue figures from quarterly PDF filings. The developer leaves the default temperature at $T=0.7$.
  * *Result:* On run 1, the model extracts `"$4.2M"`. On run 2, the model extracts `"$4.5M"`. The non-zero temperature caused stochastic sampling among closely scored numerical tokens.
  * *Engineering Mandate:* For any task where there is only one factual truth (data extraction, code generation, classification, legal compliance), **temperature MUST be set to 0.0**.
* **Marketing Copywriting Optimization:** An advertising agency needs 20 distinct taglines for a sneaker brand.
  * *Low Temp Failure:* At $T=0.0$, the model outputs `"Step into greatness"` 20 times in a row.
  * *High Temp Success:* At $T=1.5$, the model traverses the tail of its probability distribution, outputting 20 radically diverse concepts incorporating slang, evocative imagery, and varied rhythmic structures.

---

#### 4. Diagnostic Reviewer Questions & Partner Lens Exercises

##### Diagnostic Question 1: Universal Temperature Rule
*Formulate a single engineering decision rule for selecting temperature in production systems.*
> **Decision Rule:** *"If the task has a single verifiable ground-truth answer (classification, entity extraction, mathematical calculation, code compilation), set $T = 0.0$; if the task benefits from diversity, novelty, and multiple valid expressions (brainstorming, creative writing, stylistic paraphrasing), set $T \ge 1.0$."*

##### Structural Lens Audit (Creative Divergence Analysis)
* **Analytical Focus:** Inspect the 5 runs at $T=0.0$ versus the 5 runs at $T=1.8$ on `creative_prompt_duo`.
* **Diagnostic Audit:**
  1. Did the 5 outputs at $T=0.0$ produce verbatim identical text, or did minor token variations occur?
  2. At $T=1.8$, did the outputs exhibit genuine lexical diversity, or did the model degrade into nonsensical syntax?
  3. *Key Takeaway for Reflection:* High temperature expands the creative frontier by sampling low-probability, high-surprise tokens, but it operates on the brink of structural disintegration.

##### Semantic Lens Audit (Classification Resilience Analysis)
* **Analytical Focus:** Inspect the 5 runs of `classifier_high` on `sample_msg`.
* **Diagnostic Audit:**
  1. Did the predicted label flip to an incorrect category on any of the 5 runs at $T=1.8$?
  2. If the label never flipped, explain why in terms of **logit separation** rather than assuming the model "ignored" the temperature parameter.
  3. *Key Takeaway for Reflection:* A stable label under high temperature does not mean temperature was inactive; it proves the input example sat deeply within the category cluster with overwhelming logit dominance.

---

#### 5. Mathematical & Behavioral Sampling Matrix: Temperature Spectra

```
┌─────────────┬───────────────────────────┬───────────────────────────┬────────────────────────────────┐
│ TEMP ($T$)  │ MATHEMATICAL TRANSFORMATION│ BEHAVIORAL MANIFESTATION  │ PRODUCTION USE CASES           │
├─────────────┼───────────────────────────┼───────────────────────────┼────────────────────────────────┤
│ $T = 0.0$   │ Softmax approaches argmax;│ Completely deterministic; │ JSON extraction, code syntax,  │
│             │ Entropy collapses to 0    │ 100% reproducible results │ classification, SQL generation │
├─────────────┼───────────────────────────┼───────────────────────────┼────────────────────────────────┤
│ $T = 0.7$   │ Standard default scaling; │ Balanced fluency with     │ Conversational chatbots,       │
│             │ moderate entropy          │ natural stylistic variance│ general summarization          │
├─────────────┼───────────────────────────┼───────────────────────────┼────────────────────────────────┤
│ $T = 1.8$   │ Logit gaps compressed;    │ Highly stochastic; samples│ Creative fiction, poetry,      │
│             │ Softmax approaches uniform│ distribution tail; flips  │ novel product naming           │
└─────────────┴───────────────────────────┴───────────────────────────┴────────────────────────────────┘
```

---

## MODULE 05: SYNTHESIS, METACOGNITIVE REFLECTION & RUBRIC MASTERY
### Laboratory Section: `9.5 Part 5: Individual Reflections` & `9.6-9.7 Submission Guide`
### Focus: *Translating Empirical Metrics into Grounded Engineering Insights*

```
Reflection Construction Architecture:
[Empirical Data Point]  ──▶  [Underlying Mechanism]  ──▶  [Production Boundary]
(Real numbers from lab)       (Why the LLM did it)         (What you will never trust)
```

---

#### 1. Academic Context & Metacognitive Framework

##### The Empirical Reflection Standard (Rubric Section 9.5 & 9.7)
The grading rubric explicitly awards **15 points** for individual reflections and strictly penalizes general, non-empirical platitudes.
* **The Unacceptable Generalized Statement:** *"I learned that prompt engineering is important and that few-shot helps the model be more accurate, and high temperature is creative."* $\to$ **Score: 0/15 points.**
* **The Grounded Empirical Statement:** *"In Part 2, zero-shot achieved 75% accuracy (6/8), failing on Partner A's second edge-case message by outputting a format error ('Label: urgent'). Adding 4 exemplars in few-shot resolved the format error, raising accuracy to 100% (8/8). However, this expanded prompt tokens from 52 to 248 tokens—a 376% increase. In production, I would not trust a raw prompt to enforce exact word counts, as our V3 experiment drifted by 8 words."* $\to$ **Score: 15/15 points.**

##### The Three Core Inquiries of the Reflection
Each partner must independently author between **80 and 120 words** addressing two foundational prompts:
1. **The Maximum Variance Factor:** Across the four activities (Anatomy Ladder, Few-Shot Classifier, System Instructions, Temperature), what specific engineering intervention produced the most profound shift in the model's output?
2. **The Inviolable Boundary of Trust:** Identify one task or constraint that you would **never trust a raw prompt to execute reliably in production** without external programmatic enforcement (e.g., regex, Python validation, deterministic APIs).

---

#### 2. Rubric Mastery & Pre-Submission Audit Checklist

Before submitting the shared Jupyter Notebook, both partners should execute the following comprehensive pre-flight verification:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          PRE-SUBMISSION VERIFICATION AUDIT                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ ] 1. STUDENT ATTRIBUTION                                                          │
│     Student attribution names are clearly recorded in Cell 0.             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ ] 2. PART 1: THE ANATOMY LADDER (25 Points)                                          │
│     • V1, V2, and V3 completed as a continuous, coherent ladder.                       │
│     • V3 contains at least ONE checkable length/format constraint AND at least ONE      │
│       checkable tone/vocabulary constraint.                                            │
│     • Both `check()` calls populated with real values and executed.                    │
│     • Partner A reflection addresses length/format lens.                               │
│     • Partner B reflection addresses tone/vocabulary lens.                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ ] 3. PART 2: ZERO-SHOT VS. FEW-SHOT CLASSIFIER (30 Points)                           │
│     • Exactly 8 test messages authored (4 by Partner A, 4 by Partner B).               │
│     • At least 3 distinct labels chosen in `lab_labels`.                               │
│     • 3 to 5 few-shot exemplars provided in `lab_examples`.                            │
│     • ZERO DATA LEAKAGE: Automated assertion in Cell 16 passes cleanly.                │
│     • Both `evaluate()` tables printed with accuracy scores and per-author breakdown.  │
│     • Token counts printed for zero-shot vs few-shot.                                  │
│     • Partner A reflection analyzes per-author difficulty disparity.                   │
│     • Partner B reflection classifies an error (content vs format) and evaluates ROI.  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ ] 4. PART 3: SYSTEM INSTRUCTION DUEL (15 Points)                                     │
│     • Fictional `BUSINESS_NAME` and `SHARED_CUSTOMER_MESSAGE` specified.               │
│     • `system_angry` written: Harsh tone, but preserves all correct substantive steps. │
│     • `system_nice` written: Apologetic tone, preserving identical substantive steps.  │
│     • Partner A reflection evaluates angry reply (did tone eat substance?).            │
│     • Partner B reflection evaluates nice reply (substantive difference vs padding?).  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ ] 5. PART 4: TEMPERATURE DUEL (15 Points)                                            │
│     • `creative_prompt_duo` run 5 times at T=0.0 and 5 times at T=1.8 (or 1.5–2.0).    │
│     • Classifier run 5 times at T=0.0 and 5 times at high temperature.                 │
│     • Partner A reflection analyzes creative diversity vs repetition.                  │
│     • Partner B reflection analyzes classifier flipping and logit dominance.          │
│     • Concrete temperature decision rule articulated.                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ ] 6. PART 5: INDIVIDUAL REFLECTIONS (15 Points)                                      │
│     • Two distinct reflections (80–120 words each).                                    │
│     • Grounded in specific numerical results and output artifacts from the notebook.   │
│     • Identifies what changed outputs most and one thing never to trust a prompt to do.│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [ ] 7. RUNTIME ENVIRONMENT HYGIENE                                                     │
│     • All cells executed sequentially from top to bottom with visible outputs.         │
│     • No API keys printed or committed; `.env` kept strictly local.                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## CONSOLIDATED TECHNICAL REFERENCE & LANGCHAIN API COMPENDIUM

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LANGCHAIN & GOOGLE GENAI SYNTAX REFERENCE                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. INITIALIZING THE GEMINI CHAT MODEL                                                  │
│    from langchain_google_genai import ChatGoogleGenerativeAI                           │
│    llm = ChatGoogleGenerativeAI(                                                       │
│        model="gemini-3.1-flash-lite",                                                  │
│        google_api_key=api_key,                                                         │
│        temperature=0.0                                                                 │
│    )                                                                                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. CONSTRUCTING CHAT PROMPTS                                                           │
│    from langchain_core.prompts import ChatPromptTemplate                               │
│    prompt = ChatPromptTemplate.from_messages([                                         │
│        ("system", "You are an expert technical editor."),                              │
│        ("human", "Analyze the following text: {input_text}")                          │
│    ])                                                                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. FEW-SHOT IN-CONTEXT CHAT PROMPTING                                                  │
│    from langchain_core.prompts import FewShotChatMessagePromptTemplate                 │
│    example_turn = ChatPromptTemplate.from_messages([                                   │
│        ("human", "Message: {message}"),                                                │
│        ("ai", "{label}")                                                               │
│    ])                                                                                  │
│    few_shot_prompt = ChatPromptTemplate.from_messages([                                │
│        ("system", SYSTEM_RULES),                                                       │
│        FewShotChatMessagePromptTemplate(example_prompt=example_turn, examples=data),   │
│        ("human", "Message: {message}")                                                 │
│    ])                                                                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. DECLARATIVE LCEL PIPELINE WITH PARSER                                               │
│    from langchain_core.output_parsers import StrOutputParser                           │
│    chain = prompt | llm | StrOutputParser()                                            │
│    result = chain.invoke({"input_text": "Sample text"})                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. ACCURATE TOKEN AUDITING                                                             │
│    rendered = prompt.invoke({"message": "test"}).to_string()                           │
│    token_count = llm.get_num_tokens(rendered)                                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## CONSOLIDATED ACADEMIC BIBLIOGRAPHY & UPSTREAM DOCUMENTATION

1. **Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020).** *Language Models are Few-Shot Learners.* Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 1877–1901. [Foundational paper establishing in-context learning and exemplar scaling].
2. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** *Attention Is All You Need.* Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998–6008. [Foundational architecture of multi-head self-attention mechanisms].
3. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022).** *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* Advances in Neural Information Processing Systems (NeurIPS 2022), 35, 24824–24837.
4. **Google DeepMind. (2024).** *Gemini: A Family of Highly Capable Multimodal Models.* Technical Report, Google Research. [Architecture, tokenization, and instruction-tuning specifications of Gemini models].
5. **Anthropic. (2023).** *Prompt Engineering Interactive Tutorial and Best Practices.* Anthropic Research Documentation. [System prompt isolation, role prompting, and negative constraint mitigation].
6. **Chase, H. (2023).** *LangChain: Building applications with LLMs through composability.* Official Documentation and Architecture Guides. [LCEL protocol, runnable interfaces, and few-shot prompt serialization].
