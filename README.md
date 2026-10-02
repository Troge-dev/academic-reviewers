# BS Data Science 3rd Year: Academic Keynote & Reviewer Suite

[![Live Reviewer Portal](https://img.shields.io/badge/Live_Portal-GitHub_Pages-2DD4BF?style=for-the-badge&logo=github)](https://troge-dev.github.io/academic-reviewers/)
[![Disciplines](https://img.shields.io/badge/Disciplines-3_Courses-E5A967?style=for-the-badge)](https://troge-dev.github.io/academic-reviewers/)
[![Interactive Keynotes](https://img.shields.io/badge/Keynote_Decks-11_Interactive_Decks-E0533C?style=for-the-badge)](https://troge-dev.github.io/academic-reviewers/)
[![Presentation Slides](https://img.shields.io/badge/Slides-140+_Interactive-4A90E2?style=for-the-badge)](https://troge-dev.github.io/academic-reviewers/)

> **Live Interactive Keynote Hub:**  
> 🔗 **[https://troge-dev.github.io/academic-reviewers/](https://troge-dev.github.io/academic-reviewers/)**

---

## 🏛️ Curriculum Topology & Reviewer Suite

This repository serves as the centralized academic repository for **BS Data Science 3rd Year** courses and companion General Education subjects. Every reviewer is delivered in two complementary, high-retention formats:
1. **Interactive 3D Flip-Card Keynote Decks (`.html`)**: Single-file standalone HTML5 presentations with 3D card flips, luminous dark mode, edge-hover controls, and zero dependencies.
2. **Four-Tier Academic Study Guides (`.md`)**: Comprehensive theoretical textbooks structured into Theoretical Deep Dives, Key Definitions, Applied Failure Modes, and Diagnostic Oral Defense Q&As.

---

### 📚 Course Index & Direct Links

| Course | Subject Focus | Keynote Presentations | Companion Guides |
| :--- | :--- | :--- | :--- |
| **DS312** | **Data Mining & Applications** | <ul><li>[DS312 Master Synthesis Deck](dma/DS312_Master_Synthesis_Keynote_Deck.html) (18 slides)</li><li>[Modules 1–3 Foundations & Techniques](dma/DS312_Module1_to_3_Foundations_and_Techniques.html) (14 slides)</li><li>[Module 4 Structured Data & EDA](dma/DS312_Module4_Structured_Data_and_EDA.html) (16 slides)</li><li>[Module 5 Clinical NLP & Text Mining](dma/DS312_Module5_Unstructured_Text_Mining_and_NLP.html) (16 slides)</li><li>[Module 6 Supervised vs Unsupervised](dma/DS312_Module6_Supervised_vs_Unsupervised_Learning.html) (16 slides)</li></ul> | <ul><li>[DS312 Master Academic Reviewer Textbook](dma/DS312_Master_Academic_Reviewer_and_Companion_Guide.md)</li><li>[Module 6 Theoretical Companion Guide](dma/DS312_Module6_Supervised_vs_Unsupervised_Learning_Companion_Guide.md)</li></ul> |
| **DS314** | **Generative AI & LLM Systems** | <ul><li>[Topics 1–3 Master Synthesis Deck](genai/DS314_Topics1_to_3_Master_Interactive_Presentation.html) (15 slides)</li><li>[Topic 1 Foundations & Sampling](genai/DS314_Topic1_Foundations_Interactive_Presentation.html) (8 slides)</li><li>[Topic 2 Advanced RAG & Vector DBs](genai/DS314_Topic2_RAG_Systems_Interactive_Presentation.html) (8 slides)</li><li>[Topic 3 LangChain & Autonomy](genai/DS314_Topic3_LangChain_Autonomy_Interactive_Presentation.html) (8 slides)</li><li>[Week 4 Prompt Engineering Deck](genai/DS314_Week4_Prompt_Engineering_Interactive_Presentation.html) (9 slides)</li></ul> | <ul><li>[Topics 1–3 Ultimate Companion Guide](genai/DS314_Topics1_to_3_Ultimate_Companion_Guide.md)</li><li>[Week 4 Prompt Engineering Guide](genai/DS314_Week4_Prompt_Engineering_Companion_Guide.md)</li></ul> |
| **GE** | **Gender & Society** | <ul><li>[Chapter 6: Architecture of GBV Presentation](gbv/index.html) (12 slides)</li></ul> | <ul><li>[GBV Speaker Companion & Study Guide](gbv/gbv_speaker_companion_and_study_guide.md)</li><li>[Executive Slide Deck Spec](gbv/gbv_executive_slide_deck.md)</li></ul> |

---

## 🎮 Interactive Keynote Navigation & Controls

All interactive presentations feature zero-dependency vanilla JS and responsive CSS:

* **Slide Progression:** `→` or `PageDown` (Next) | `←` or `PageUp` (Previous)
* **3D Card Flip (Active Recall):** Click any individual card, or press `Space` / `Enter`
* **Toggle Controls & Footer:** `C` (Reveals/collapses slide scrubber and navigation pills)
* **Toggle Color Theme:** `T` (Switches between Warm Linen Alabaster and Deep Space Slate)
* **Hover Edge Reveals:** Controls and toggle pills are invisible during reading and reveal smoothly when the mouse moves near the top (0–65px) or bottom edges.

---

## 🏗️ Repository Architecture

```text
academic-reviewers/
├── index.html                                        # Master Central Hub & Subject Search
├── README.md                                         # Repository Documentation & Directory
│
├── dma/                                              # DS312: Data Mining & Applications
│   ├── index.html                                    # DMA Dedicated Portal
│   ├── DS312_Master_Synthesis_Keynote_Deck.html       # 18-Slide Grand Keynote Deck
│   ├── DS312_Module1_to_3_Foundations_and_Techniques.html
│   ├── DS312_Module4_Structured_Data_and_EDA.html
│   ├── DS312_Module5_Unstructured_Text_Mining_and_NLP.html
│   ├── DS312_Module6_Supervised_vs_Unsupervised_Learning.html
│   ├── DS312_Master_Academic_Reviewer_and_Companion_Guide.md
│   └── DS312_Module6_Supervised_vs_Unsupervised_Learning_Companion_Guide.md
│
│
├── genai/                                            # DS314: Generative AI & LLM Systems
│   ├── index.html                                    # GENAI Dedicated Portal
│   ├── DS314_Topics1_to_3_Master_Interactive_Presentation.html
│   ├── DS314_Topic1_Foundations_Interactive_Presentation.html
│   ├── DS314_Topic2_RAG_Systems_Interactive_Presentation.html
│   ├── DS314_Topic3_LangChain_Autonomy_Interactive_Presentation.html
│   ├── DS314_Week4_Prompt_Engineering_Interactive_Presentation.html
│   ├── DS314_Week4_Prompt_Engineering_Companion_Guide.md
│   └── DS314_Topics1_to_3_Ultimate_Companion_Guide.md
│
└── gbv/                                              # Gender & Society: GBV Architecture
    ├── index.html                                    # GBV 3D Interactive Presentation Deck
    ├── gbv_executive_presentation.html               # Presentation Mirror
    ├── gbv_speaker_companion_and_study_guide.md      # Oral Defense Guide
    └── gbv_executive_slide_deck.md                   # Slide Specification Blueprint
```

---

## 💻 Tech Stack & Design System

- **Fonts:** Space Grotesk (Headlines), Plus Jakarta Sans (Editorial Body), JetBrains Mono (Technical Kicking & Metadata).
- **CSS Architecture:** Reactive CSS custom properties with WCAG AA compliance in both Light and Obsidian Dark modes.
- **Card Topology:** CSS 3D Transforms (`perspective: 1000px`, `transform-style: preserve-3d`, `rotateY(180deg)`).
- **Zero Build Step:** 100% pure client-side standard HTML5/CSS3/ES6 running directly from any web browser or GitHub Pages CDN.
