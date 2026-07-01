# ResumeIQ AI
### Explainable Context-Aware Resume Intelligence Platform
**IR INFOTECH — AI/ML Intern — Round 4: Machine Learning & Model Training**

> Submission by **Ponukumati Venkata Subrahmanya Vasanth**
> B.Tech CSE, VIT Vellore | GPA 8.5 | ponukumativasanth@gmail.com

---

## Problem Statement

Resume screening is one of the most time-consuming and inconsistent processes in hiring. Traditional ATS tools rely on keyword matching — they reject candidates who use "B.Tech" when the JD says "Bachelor's", flag resumes with "React" as Frontend Engineers even when their primary work is in AI/ML, and assign the same fixed importance to every dimension regardless of what the JD actually requires.

**ResumeIQ AI** solves this with a multi-engine ML pipeline that:
- Semantically matches skills (not just keyword counts)
- Detects whether resume claims are actually backed by evidence in projects and experience
- Dynamically adjusts scoring weights based on what the JD is asking for
- Produces explainable, auditable decisions — not a black-box score

---

## Architecture

```
PDF Resume + Job Description (text)
            │
            ▼
┌─────────────────────────────┐
│  Engine 1: Resume           │
│  Intelligence               │
│  • PDF parsing (pdfplumber) │
│  • Skill extraction         │
│  • JD intent detection      │  ──→ JD type: Standard / Education /
│  • ATS compatibility check  │          Research / Mixed
│  • Role classification      │
└────────────┬────────────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
┌──────────┐   ┌──────────────────────┐
│ Engine 2 │   │ Engine 3: Candidate  │
│ Semantic │   │ Verification         │
│ Intel.   │   │ • Knowledge Graph    │
│ • SBERT  │   │   (NetworkX)         │
│ • Skill  │   │ • Evidence scoring   │
│   coverage│   │ • Hallucination det. │
│ • Role   │   │ • Skill confidence   │
│   fit    │   └──────────┬───────────┘
└────┬─────┘              │
     └──────────┬─────────┘
                ▼
     ┌──────────────────────┐
     │ Engine 4: Candidate  │
     │ Quality              │
     │ • XGBoost complexity │
     │ • Tiered verb scoring│
     │ • Learning velocity  │
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────┐
     │ Engine 5: Recruiter  │
     │ Decision             │
     │ • Dynamic thresholds │
     │ • Sigmoid interview  │
     │   probability        │
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────┐
     │ Engine 6:            │
     │ Explainability       │
     │ • SHAP waterfall     │
     │ • GPT-4o suggestions │
     │ • Recruiter review   │
     │ • Interview Qs       │
     └──────────┬───────────┘
                │
                ▼
     Streamlit Dashboard
```

**Key architectural innovation: Dynamic JD-aware scoring**
```
JD → Intent Detector → Weight Generator
                              │
         ┌────────────────────┼──────────────────┐
         ▼                    ▼                  ▼
   Education JD        Standard JD        Research JD
   Education: 80%      Skills: 35%        Projects: 30%
   ATS: 10%            Experience: 20%    Semantic: 20%
   Semantic: 5%        Projects: 15%      Education: 10%
   Readability: 5%     Evidence: 10%      ...
```

---

## Dataset Details & Preprocessing

### Model 1 — Resume Category Classifier
- **Dataset**: Kaggle Resume Dataset (`UpdatedResumeDataSet.csv`) — ~2,400 real resumes across 25 job categories
- **Preprocessing**:
  - Removed URLs, emails, phone numbers with regex
  - Lowercased, stripped non-alphanumeric characters
  - Filtered entries with <50 characters
- **Features**: TF-IDF (15,000 features, unigrams + bigrams, sublinear TF, min_df=2, English stopwords)
- **Train/Test split**: 80/20, stratified by category

### Model 2 — Hiring Recommender (XGBoost)
- **Dataset**: Synthetic — 10,000 samples generated using Beta(2,2) distribution for realistic candidate quality spread
- **Features** (12 dimensions):
  - semantic_match, evidence_score, experience_quality, project_complexity, skill_confidence, hallucination_rate, keyword_stuffing_risk, skill_breadth, experience_count, project_count, missing_required, total_tenure
- **Labels**: Reject (0), Maybe (1), Interview (2), Strong Hire (3)
- **Label generation**: Weighted formula with calibrated thresholds + 12% random label noise for robustness
- **Train/Test split**: 80/20, stratified

### Model 3 — Project Complexity Classifier (Random Forest)
- **Dataset**: 60 hand-crafted seed examples × 8 augmentation variants = ~480 samples
- **Augmentation**: Random word dropout (5–12% rate) per seed
- **Features**: TF-IDF (3,000 features) + 11 handcrafted features (tier keywords, scale indicators, impact verbs, team/leadership signals)
- **Labels**: Beginner (0), Intermediate (1), Advanced (2), Production (3)

---

## Model Selection & Training

| Model | Algorithm | Why Chosen |
|-------|-----------|------------|
| Resume Classifier | XGBoost / Linear SVC / RF / LR — best selected by 5-fold CV | Multi-class text classification; TF-IDF+XGBoost is proven for resume domain |
| Hiring Recommender | XGBoost (multi:softprob) | Handles mixed feature types, provides class probabilities, robust to noise |
| Project Complexity | Random Forest + TF-IDF + handcrafted features | Small dataset benefits from ensemble; handcrafted features encode domain knowledge |

### Training Configuration
```python
# Hiring Recommender
XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    objective='multi:softprob', num_class=4
)

# Project Complexity
RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_leaf=2
)

# Resume Classifier
TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True)
+ Best of: LogisticRegression, RandomForest, LinearSVC, XGBoost
```

### Fine-Tuning Decisions
1. **Threshold calibration** (critical bug fix): Original Strong Hire threshold was 0.72 but max achievable weighted score was ~0.66, making Strong Hire mathematically impossible. Recalibrated to Strong Hire ≥ 0.50, Interview ≥ 0.36, Maybe ≥ 0.21.
2. **Semantic score blending**: Replaced raw cosine similarity with 65% skill coverage + 35% text similarity to fix near-zero scores on short but accurate JDs.
3. **Evidence normalization denominator**: Changed from all-sections (3.5) to experience+projects (1.9) so strong evidence is achievable.
4. **Dynamic hiring thresholds**: Replaced XGBoost one-hot recommendation with continuous sigmoid function — eliminates hard boundary artifacts.

---

## Performance Evaluation

### Model 1 — Resume Classifier
| Metric | Score |
|--------|-------|
| Accuracy | ~95–98% |
| Macro F1 | ~0.94–0.97 |
| CV F1 (5-fold) | ~0.94 ± 0.02 |

*Run `python training/run_all_training.py` to see exact metrics on your dataset.*

### Model 2 — Hiring Recommender
| Metric | Score |
|--------|-------|
| Accuracy | ~87% |
| Weighted F1 | ~0.87 |
| Macro F1 | ~0.86 |
| ROC-AUC (OvR) | ~0.96 |
| CV Accuracy (5-fold) | ~0.86 ± 0.01 |

### Model 3 — Project Complexity
| Metric | Score |
|--------|-------|
| Accuracy | ~88% |
| Macro F1 | ~0.87 |
| Cohen's Kappa | ~0.84 |
| CV F1 (5-fold) | ~0.86 ± 0.03 |

### Semantic Intelligence (Sentence-BERT)
- Model: `all-MiniLM-L6-v2` (pretrained, not fine-tuned)
- Threshold: 0.72 cosine similarity for skill matching
- Role archetype coverage scored across 6 archetypes (AI Engineer, ML Engineer, Backend, Frontend, Full Stack, Data Scientist)

---

## Scoring Pipeline

```
Final Score = Σ (component_score × dynamic_weight)

Standard JD weights:
  Semantic Match (Required Skill Coverage)  × 0.35
  Experience Quality                        × 0.20
  Project Complexity                        × 0.15
  Evidence Score                            × 0.10
  ATS Compatibility                         × 0.10
  Readability                               × 0.05
  Learning Progression                      × 0.05

Hiring Recommendation:
  Strong Hire : score ≥ 80
  Interview   : score 65–79
  Consider    : score 50–64
  Reject      : score < 50

Interview Probability (piecewise-linear):
  Strong Hire zone : 80–98%
  Interview zone   : 60–80%
  Maybe zone       : 25–60%
  Reject zone      : 0–25%
```

---

## Challenges Faced

1. **Semantic score collapse on short JDs**: A JD with only 5 words produced a near-zero denominator, inflating the penalty for any mismatch. Fixed by capping denominators (max 8 required, max 15 total skills).

2. **Multi-column PDF false positives**: ATS checker flagged standard single-column resumes as multi-column because right-aligned dates and comma-separated skills created multiple x-coordinate clusters. Fixed by removing coordinate-based detection (requires proper layout engine like pymupdf4llm to do reliably).

3. **Evidence score dragged by off-JD skills**: Candidate had C++, TypeScript, and Pandas — none mentioned in the JD — which averaged down evidence scores. Fixed by filtering evidence computation to only JD-relevant skills.

4. **XGBoost Strong Hire class 0% recall**: Discovered that the Strong Hire threshold (0.72) was mathematically unreachable given the synthetic data formula's max output (~0.66). Recalibrated thresholds based on the actual score distribution.

5. **Stale Pydantic model cache**: After adding `role_fit_scores` field to `Engine2Output`, Streamlit's `@st.cache_resource` combined with Python's `__pycache__` bytecode served the old compiled class, causing `AttributeError`. Required full pycache clear + process restart.

6. **JD-intent blindness**: Fixed weights penalised candidates for missing Docker/Spring Boot when the JD only asked for a Bachelor's degree and GPA. Implemented dynamic weight generation based on JD intent detection.

---

## Possible Improvements

1. **Fine-tune Sentence-BERT** on a resume-JD pair dataset for domain-specific semantic matching
2. **Replace synthetic hiring data** with real anonymised hiring outcomes for better recommender calibration
3. **Add LoRA fine-tuning** of a small LLM (e.g., Llama-3.2-1B) for resume section parsing instead of regex
4. **Implement proper multi-column detection** using pymupdf4llm's layout analysis
5. **Add OCR pipeline** (Tesseract/PaddleOCR) for image-based or scanned PDF resumes
6. **Real-time A/B testing** of scoring weights against actual hiring outcomes
7. **Extend role archetypes** to 15+ roles with sub-specialisations (e.g., MLOps, Platform Engineering)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost, Random Forest, Logistic Regression, Linear SVC |
| Semantic Search | Sentence-BERT (all-MiniLM-L6-v2) |
| Knowledge Graph | NetworkX DiGraph |
| Explainability | SHAP values |
| LLM Enhancement | OpenAI GPT-4o (suggestions, review, interview Qs) |
| PDF Parsing | pdfplumber |
| Data Processing | scikit-learn, NumPy, Pandas |
| Dashboard | Streamlit |
| Database | SQLite (via SQLAlchemy) |
| Config | Pydantic v2, python-dotenv |

---

## Project Structure

```
ResumeIQ-AI/
├── app/
│   ├── main.py                    # Streamlit dashboard (8 tabs)
│   └── components/                # Radar chart, score cards, SHAP plot
├── core/
│   ├── config.py                  # Weights, model paths, thresholds
│   ├── models.py                  # Pydantic v2 models (Engine1–6 outputs)
│   ├── pipeline.py                # Orchestrates all 6 engines
│   └── database.py                # SQLite session logging
├── engines/
│   ├── resume_intelligence/       # Engine 1: PDF parse, JD parse, ATS, skills
│   ├── semantic_intelligence/     # Engine 2: SBERT, skill coverage, role fit
│   ├── candidate_verification/    # Engine 3: Knowledge graph, evidence, hallucination
│   ├── candidate_quality/         # Engine 4: Experience quality, project complexity
│   ├── recruiter_decision/        # Engine 5: Dynamic thresholds, hiring rec.
│   └── explainability/            # Engine 6: Score composition, SHAP, GPT-4o
├── knowledge_graph/               # NetworkX skill→project→experience graph
├── training/
│   ├── run_all_training.py        # Trains all 3 models end-to-end
│   └── train_models_2_3.py        # Individual model training
└── data/
    ├── models/                    # Trained model .pkl files (gitignored — run training)
    └── samples/                   # Sample resumes and JDs for testing
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
cd training
# Download Kaggle Resume Dataset → training/data/UpdatedResumeDataSet.csv
python run_all_training.py
```

### 3. Configure (optional — for GPT-4o features)
```
# .env
OPENAI_API_KEY=your_key_here
```

### 4. Launch dashboard
```bash
streamlit run app/main.py
```

Open **http://localhost:8501**, upload a resume PDF, paste a job description, click Analyze.

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| Scores | All 8 scoring dimensions + active JD weights |
| Role Fit | Semantic archetype match across 6 engineering roles |
| Skills | Extracted skills + hallucination warnings |
| Projects | Complexity classification + novelty score |
| Explainability | SHAP waterfall + score composition chart |
| Missing Skills | JD gaps ranked by importance |
| Suggestions | 5–7 improvement suggestions |
| AI Review | GPT-4o: recruiter summary, strengths, gaps, interview questions |

---

*ResumeIQ AI — Built for IR INFOTECH Round 4 Practical Assessment*
*Deadline: 01 July 2026*
