# ResumeIQ AI — IR INFOTECH Practical Assessment Submission

**Candidate:** Ponukumati Venkata Subrahmanya Vasanth  
**Position:** AI / ML Intern  
**Round:** Round 4 — Machine Learning & Model Training  

---

## 1. Problem Statement

Traditional Applicant Tracking Systems (ATS) use keyword matching and fail to understand
candidate competency, evidence quality, project complexity, or provide explainable scores.

**ResumeIQ AI** is an Explainable, Context-Aware Resume Intelligence Platform that:
- Parses and understands resume content semantically
- Validates whether claimed skills are backed by evidence
- Predicts hiring outcomes with full explainability (SHAP)
- Ranks multiple candidates against a job description

**Three ML Models Trained:**
| # | Model | Task |
|---|---|---|
| 1 | Resume Category Classifier | Predict job role from resume text (25 classes) |
| 2 | Hiring Recommender (XGBoost) | Predict Reject/Maybe/Interview/Strong Hire |
| 3 | Project Complexity Classifier | Predict Beginner/Intermediate/Advanced/Production |

---

## 2. Dataset Details & Preprocessing

### Model 1 — Resume Classifier
- **Dataset:** Kaggle Resume Dataset (`snehaanbhawal/resume-dataset`)
- **URL:** https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- **Size:** 2,400+ labelled resumes, 24 job categories
- **Columns:** `ID`, `Resume_str`, `Resume_html`, `Category`
- **Preprocessing:**
  - Removed URLs, email addresses, phone numbers
  - Lowercased, removed special characters
  - TF-IDF vectorization: 15,000 features, unigrams + bigrams, sublinear TF scaling
- **Split:** 80/20 stratified train/test

### Model 2 — Hiring Recommender
- **Dataset:** 10,000 synthetically generated feature vectors
- **Features:** 12 numerical features (semantic match, evidence score, experience quality, etc.)
- **Label Generation:** Rule-based scoring with 12% label noise (realistic hiring uncertainty)
- **Split:** 80/20 stratified

### Model 3 — Project Complexity
- **Dataset:** 1,200 labeled project descriptions (60 seed samples × 8x text augmentation)
- **Features:** TF-IDF (3,000 features) + 11 handcrafted features (metrics, deployment signals, team signals)
- **Split:** 80/20 stratified

---

## 3. Solution Approach

### Architecture: 7-Engine Pipeline

```
Resume PDF + Job Description
        ↓
ENGINE 1: Resume Intelligence
  ├── PDF Parser (pdfplumber)
  ├── Skill Extractor (spaCy EntityRuler + custom taxonomy)
  ├── ATS Checker (rule-based)
  ├── Keyword Stuffing Detector (heuristic)
  └── Resume Classifier (ML Model 1) ← TRAINED
        ↓
Knowledge Graph (NetworkX)
  skill ↔ project ↔ experience edges
        ↓
ENGINE 2: Semantic Intelligence
  └── Sentence-BERT cosine similarity (all-MiniLM-L6-v2)
        ↓
ENGINE 3: Candidate Verification
  ├── Evidence Validator (cross-section search + graph)
  ├── Hallucination Detector (Isolation Forest)
  └── Skill Confidence (0.6×evidence + 0.4×semantic)
        ↓
ENGINE 4: Candidate Quality
  ├── Project Complexity (ML Model 3) ← TRAINED
  ├── Experience Quality (impact pattern regex)
  ├── Learning Progression (timeline velocity)
  └── Project Novelty (TF-IDF rarity + SBERT distance)
        ↓
ENGINE 5: Recruiter Decision
  ├── ATS Scorer (penalty-based)
  └── Hiring Recommender (ML Model 2) ← TRAINED
        ↓
ENGINE 6: Explainability
  ├── SHAP TreeExplainer (feature contributions)
  ├── Score Composer (8-dimension weighted scoring)
  └── Suggestion Generator (OpenAI GPT-4o)
        ↓
Streamlit Dashboard
```

### Weighted Scoring Formula

```
Final Score = 
  Semantic Match      × 30%
  + Evidence Score    × 20%
  + Experience Quality× 15%
  + Project Complexity× 10%
  + Skill Confidence  × 10%
  + ATS Compatibility ×  5%
  + Readability       ×  5%
  + Learning Progression× 5%
```

---

## 4. Model Selection

| Task | Model Chosen | Why |
|---|---|---|
| Resume Classification | TF-IDF + Best of {LR, RF, LinearSVC, XGBoost} | Compared 4 models, selected by 5-fold CV F1 |
| Hiring Recommendation | XGBoost Multiclass | Handles mixed features, SHAP-native, best on tabular data |
| Project Complexity | Random Forest + TF-IDF | Handles sparse+dense features, interpretable importance |
| Semantic Matching | Sentence-BERT (pre-trained) | Strong zero-shot performance, no GPU training needed |
| Hallucination Detection | Isolation Forest | Unsupervised anomaly — no labels needed |

---

## 5. Training Process

### Model 1 — Resume Classifier
1. Load Kaggle CSV (2,484 rows)
2. Clean text (URL/email/phone removal, lowercase)
3. TF-IDF fit on train set (prevent data leakage)
4. Train 4 models: LR, RF, LinearSVC, XGBoost
5. 5-fold stratified cross-validation
6. Select best by CV F1-macro
7. Final evaluation on held-out test set
8. Save as `data/models/resume_classifier.pkl`

### Model 2 — Hiring Recommender
1. Generate 10k synthetic samples with correlated features
2. Assign labels via weighted scoring rule + 12% noise
3. Train XGBoost with: 300 estimators, depth=6, lr=0.05, subsample=0.8
4. 5-fold CV for robustness check
5. SHAP TreeExplainer for feature importance
6. Save as `data/models/hiring_recommender.pkl`

### Model 3 — Project Complexity
1. Seed with 60 manually labeled project descriptions
2. Augment 8x via random word dropout
3. Combine TF-IDF (3k features) + 11 handcrafted features
4. Train Random Forest (300 trees) + LinearSVC (comparison)
5. Select best by CV F1-macro
6. Test on real project descriptions
7. Save as `data/models/project_complexity.pkl`

---

## 6. Evaluation Results

*(Fill in after running the notebooks)*

### Model 1 — Resume Classifier
| Metric | Value |
|---|---|
| Test Accuracy | 81.09% |
| Macro F1 | 0.7899 |
| Weighted F1 | 0.8053 |
| CV F1 (5-fold) | 0.7120 ± 0.0208 |
| Best Model | XGBoost (200 estimators, depth=6, lr=0.1) |

### Model 2 — Hiring Recommender (XGBoost)
| Metric | Value |
|---|---|
| Test Accuracy | 87.05% |
| Weighted F1 | 0.8553 |
| Macro F1 | 0.6401 |
| ROC-AUC (OvR) | 0.8076 |
| CV Accuracy (5-fold) | 0.8702 ± 0.0045 |

*Note: "Strong Hire" class is rare (3.2% of data) — F1 for that class is low due to class imbalance. ROC-AUC of 0.8076 reflects strong overall discrimination.*

### Model 3 — Project Complexity
| Metric | Value |
|---|---|
| Test Accuracy | 100.00% |
| Macro F1 | 1.0000 |
| Cohen's Kappa | 1.0000 |
| CV F1 (5-fold) | 0.9979 ± 0.0042 |

*Real-world prediction test: 4/4 correct (Beginner, Intermediate, Advanced, Production all correctly predicted on unseen descriptions).*

---

## 7. Challenges Faced

1. **Resume section segmentation** — Resumes have no standard format. Solved with regex-based section detection with 8 section header patterns ranked by specificity.

2. **Evidence validation without labels** — Determining whether a claimed skill has real evidence is hard to label. Solved with a cross-section search + knowledge graph approach, falling back to Isolation Forest for anomaly detection.

3. **Training data for hiring decisions** — No public dataset of ground-truth hiring decisions exists. Solved by generating realistic synthetic data with domain knowledge encoded in the label generation rule.

4. **Explainability vs accuracy** — Deep models score higher but are black boxes. Chose XGBoost specifically because SHAP TreeExplainer provides exact explanations in milliseconds.

5. **ATS diversity** — ATS systems interpret PDFs differently. Built a rule-based checker that flags the top 6 structural issues (tables, multi-columns, images, missing sections, no contact info, long bullets).

---

## 8. Possible Improvements

1. **Resume Classifier** — Fine-tune DistilBERT instead of TF-IDF for 3–5% F1 improvement on ambiguous categories.

2. **Hiring Recommender** — Collect real recruiter decisions from HR professionals to replace synthetic labels. Use LightGBM LambdaRank for better ordinal learning.

3. **Evidence Validator** — Integrate GitHub API to verify code evidence for claimed programming skills.

4. **Hallucination Detector** — Label 500 resumes manually to convert from unsupervised to supervised learning.

5. **Semantic Matching** — Fine-tune Sentence-BERT on resume-JD pairs for domain-specific embeddings.

6. **Scale** — Replace SQLite with PostgreSQL, deploy on FastAPI + Docker + Kubernetes for production.

---

## 9. Project Structure

```
ResumeIQ AI/
├── engines/
│   ├── resume_intelligence/     # Engine 1: Parser, Extractor, Classifier
│   ├── semantic_intelligence/   # Engine 2: SBERT matching
│   ├── candidate_verification/  # Engine 3: Evidence, Hallucination, Confidence
│   ├── candidate_quality/       # Engine 4: Complexity, Quality, Progression
│   ├── recruiter_decision/      # Engine 5: XGBoost, Ranker
│   └── explainability/          # Engine 6: SHAP, Score, GPT-4o
├── core/                        # Models, Config, Pipeline, Database
├── knowledge_graph/             # NetworkX resume graph
├── app/                         # Streamlit dashboard
├── training/
│   ├── 01_resume_classifier.ipynb      ← Notebook 1
│   ├── 02_hiring_recommender.ipynb     ← Notebook 2
│   └── 03_project_complexity.ipynb     ← Notebook 3
└── data/
    ├── skill_taxonomy.json      # 200+ skills with aliases
    └── models/                  # Saved .pkl files after training
```

**To run the notebooks:**
```bash
pip install -r requirements.txt
cd training
jupyter notebook
```

**To run the dashboard:**
```bash
streamlit run app/main.py
```
