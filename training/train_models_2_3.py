"""Train Model 2 (Hiring Recommender) and Model 3 (Project Complexity)."""
import sys, os, re, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             cohen_kappa_score)
import xgboost as xgb

os.makedirs('plots', exist_ok=True)
os.makedirs('../data/models', exist_ok=True)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────────────────────
# MODEL 2 — HIRING RECOMMENDER (XGBoost)
# ─────────────────────────────────────────────────────────────
print("\n[2/3] Hiring Recommender (XGBoost)")
print("-" * 50)

FEATURE_NAMES = [
    'semantic_match', 'evidence_score', 'experience_quality',
    'project_complexity', 'skill_confidence', 'hallucination_rate',
    'keyword_stuffing_risk', 'skill_breadth', 'experience_count',
    'project_count', 'missing_required', 'total_tenure',
]
LABEL_NAMES2 = ['Reject', 'Maybe', 'Interview', 'Strong Hire']
N = 10_000
np.random.seed(42)

quality = np.random.beta(2, 2, N)
clip = lambda x: np.clip(x, 0.0, 1.0)
noise = lambda s: np.random.normal(0, s, N)

X2 = np.column_stack([
    clip(quality*0.80 + noise(0.15)),
    clip(quality*0.75 + noise(0.15)),
    clip(quality*0.70 + noise(0.18)),
    clip(quality*0.65 + noise(0.20)),
    clip(quality*0.72 + noise(0.15)),
    clip((1-quality)*0.40 + noise(0.10)),
    clip((1-quality)*0.30 + noise(0.10)),
    clip(quality*0.60 + noise(0.20)),
    clip(quality*0.50 + noise(0.25)),
    clip(quality*0.55 + noise(0.20)),
    clip((1-quality)*0.50 + noise(0.15)),
    clip(quality*0.60 + noise(0.20)),
])

def assign_label(row):
    s = (row[0]*0.30 + row[1]*0.20 + row[2]*0.15 + row[3]*0.10
         + row[4]*0.10 - row[5]*0.08 - row[6]*0.05
         - row[10]*0.07 + row[11]*0.05)
    # Thresholds calibrated to the actual score range of this formula.
    # Max achievable s ≈ 0.66, so the old 0.72 threshold was unreachable.
    return 3 if s >= 0.50 else 2 if s >= 0.36 else 1 if s >= 0.21 else 0

y2 = np.array([assign_label(r) for r in X2])
noise_mask = np.random.random(N) < 0.12
y2[noise_mask] = np.random.randint(0, 4, noise_mask.sum())

print(f"  Dataset: {N} synthetic samples")
for i, n in enumerate(LABEL_NAMES2):
    print(f"    {n:12s}: {(y2==i).sum()} ({(y2==i).mean()*100:.1f}%)")

X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

model2 = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    objective='multi:softprob', num_class=4, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, verbosity=0,
)
model2.fit(X2_tr, y2_tr, eval_set=[(X2_te, y2_te)], verbose=False)

y2_pred  = model2.predict(X2_te)
y2_proba = model2.predict_proba(X2_te)
acc2  = accuracy_score(y2_te, y2_pred)
f1w2  = f1_score(y2_te, y2_pred, average='weighted')
f1m2  = f1_score(y2_te, y2_pred, average='macro')
roc2  = roc_auc_score(y2_te, y2_proba, multi_class='ovr', average='macro')
cv2   = cross_val_score(model2, X2, y2, cv=cv5, scoring='accuracy', n_jobs=-1)

print(f"\n  Accuracy      : {acc2:.4f} ({acc2*100:.2f}%)")
print(f"  Weighted F1   : {f1w2:.4f}")
print(f"  Macro F1      : {f1m2:.4f}")
print(f"  ROC-AUC (OvR) : {roc2:.4f}")
print(f"  CV Accuracy   : {cv2.mean():.4f} +/- {cv2.std():.4f}")
print(f"\n{classification_report(y2_te, y2_pred, target_names=LABEL_NAMES2, digits=4)}")

cm2 = confusion_matrix(y2_te, y2_pred)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_NAMES2, yticklabels=LABEL_NAMES2, ax=ax)
ax.set_title('Hiring Recommender - Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/02_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

imp = model2.feature_importances_
idx = np.argsort(imp)
fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#ef4444' if FEATURE_NAMES[i] in
          ['hallucination_rate', 'keyword_stuffing_risk', 'missing_required']
          else '#22c55e' for i in idx]
ax.barh([FEATURE_NAMES[i].replace('_', ' ').title() for i in idx], imp[idx], color=colors)
ax.set_title('XGBoost Feature Importance - Hiring Recommender', fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('plots/02_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

with open('../data/models/hiring_recommender.pkl', 'wb') as f:
    pickle.dump({
        'model': model2, 'feature_names': FEATURE_NAMES,
        'label_names': LABEL_NAMES2, 'accuracy': float(acc2),
        'f1_weighted': float(f1w2), 'f1_macro': float(f1m2),
        'roc_auc': float(roc2),
    }, f)
print("  Model saved -> data/models/hiring_recommender.pkl")
print(f"  RESULT: Accuracy={acc2*100:.2f}% | ROC-AUC={roc2:.4f} | F1-W={f1w2:.4f}")


# ─────────────────────────────────────────────────────────────
# MODEL 3 — PROJECT COMPLEXITY (Random Forest)
# ─────────────────────────────────────────────────────────────
print("\n\n[3/3] Project Complexity Classifier (Random Forest)")
print("-" * 50)

LABEL_NAMES3 = ['Beginner', 'Intermediate', 'Advanced', 'Production']

RAW = [
    ("Simple calculator app using Python with basic arithmetic operations", 0),
    ("Todo list application built with HTML CSS JavaScript add delete complete", 0),
    ("Student grade management system using Java with CRUD file storage", 0),
    ("Number guessing game built with Python using random module while loop", 0),
    ("Basic weather app using OpenWeatherMap API temperature conditions", 0),
    ("Library book management system with add remove search Python lists", 0),
    ("Portfolio website HTML CSS JavaScript animations contact form", 0),
    ("Snake game Python pygame library score tracking collision detection", 0),
    ("BMI calculator web app using Flask form input result display", 0),
    ("Expense tracker Python CSV file storage matplotlib chart output", 0),
    ("Rock paper scissors game computer AI random choice Python", 0),
    ("Quiz application multiple choice questions score Tkinter", 0),
    ("Currency converter real-time API web scraping Python", 0),
    ("Tic-tac-toe game two player JavaScript HTML canvas", 0),
    ("Contact book CRUD SQLite database Python", 0),
    ("E-commerce web application Django REST framework user authentication product cart payment Stripe", 1),
    ("Blog platform React frontend Node.js Express backend MongoDB JWT authentication CRUD", 1),
    ("Movie recommendation system collaborative filtering cosine similarity MovieLens 78 percent accuracy", 1),
    ("Real-time chat application WebSocket Socket.io Node.js React room messaging", 1),
    ("Social media dashboard Angular RESTful API integration PostgreSQL responsive design", 1),
    ("Sentiment analysis NLTK scikit-learn IMDB reviews 85 percent accuracy", 1),
    ("Job portal web app Spring Boot React user roles recruiter candidate resume upload search", 1),
    ("Inventory management system low-stock alerts email notifications analytics Flask PostgreSQL Chart.js", 1),
    ("Music streaming application playlist management audio player React Node.js MongoDB", 1),
    ("Spam email classifier TF-IDF Naive Bayes 92 percent precision Flask API deployed", 1),
    ("Online examination portal timer auto-submit randomised questions result analytics Django", 1),
    ("Recipe sharing platform ingredient search nutritional ratings Vue.js Firebase", 1),
    ("Ride-sharing app prototype real-time location Google Maps React Native", 1),
    ("Customer churn prediction XGBoost 88 percent ROC-AUC telecom SHAP explanations", 1),
    ("News aggregator Flutter category filtering saved articles push notifications REST APIs", 1),
    ("Distributed microservices food delivery Docker Kubernetes service mesh API gateway 10k requests per minute", 2),
    ("Machine learning pipeline AWS SageMaker real-time fraud detection 99.2 percent precision 50k transactions XGBoost auto-scaling", 2),
    ("NLP legal document analysis fine-tuned BERT 94 percent F1 NER custom legal entities", 2),
    ("Real-time recommendation Redis caching Apache Kafka event streaming 500k users", 2),
    ("Computer vision defect detection manufacturing YOLOv8 custom dataset 96 percent mAP edge devices", 2),
    ("Multi-cloud infrastructure Terraform Ansible CI/CD GitHub Actions blue-green deployment zero-downtime", 2),
    ("Conversational AI RAG LangChain Pinecone OpenAI GPT-4 enterprise knowledge base 95 percent query resolution", 2),
    ("Graph neural network social network analysis PyTorch Geometric 10 million nodes", 2),
    ("Blockchain supply chain Ethereum smart contracts Solidity IPFS React MetaMask", 2),
    ("Time series forecasting energy consumption LSTM Prophet 94 percent accuracy FastAPI Grafana monitoring", 2),
    ("Federated learning privacy-preserving medical image classification 5 hospitals PySyft", 2),
    ("Search engine TF-IDF semantic search Sentence-BERT Elasticsearch 1M documents", 2),
    ("LLM fine-tuning QLoRA domain-specific customer service RLHF feedback loop", 2),
    ("Multi-agent reinforcement learning autonomous trading risk management 23 percent annual returns", 2),
    ("Real-time video analytics OpenCV DeepSORT YOLO 30fps sub-100ms latency GPU cluster", 2),
    ("Built scaled ride-sharing platform 2 million active users Django microservices PostgreSQL Redis Kubernetes AWS 99.99 percent uptime SLA", 3),
    ("Led ML-powered credit scoring fintech 100k applications daily XGBoost 96 percent AUC SHAP regulatory compliance", 3),
    ("Architected real-time data pipeline 5TB per day Kafka Spark Streaming Airflow Snowflake sub-second latency", 3),
    ("LLM document intelligence legal firm automating contract review 70 percent reduction 500 lawyers 99.5 percent availability", 3),
    ("Recommendation engine e-commerce 10 million users 35 percent click-through increase real-time collaborative filtering A/B testing", 3),
    ("Computer vision quality control automotive 99.7 percent recall 1000 parts per hour 12 production lines", 3),
    ("Healthcare analytics 50 million patient records HIPAA end-to-end encryption federated queries 18 percent ICU reduction", 3),
    ("Multi-tenant SaaS analytics GCP 10 billion events per day BigQuery Dataflow Pub/Sub 99.95 percent SLA 40 percent cost saving", 3),
    ("Led team 8 engineers fraud detection payments 60 percent reduction graph neural networks 2M transactions per hour", 3),
    ("Open-sourced MLOps framework 200 companies model drift detection retraining pipelines PyTorch TensorFlow", 3),
    ("Globally distributed content delivery 50M concurrent viewers adaptive bitrate CDN sub-2-second startup", 3),
    ("Autonomous drone fleet real-time path planning obstacle avoidance GPS 10000 acres per day", 3),
    ("Enterprise search Elasticsearch ML ranking 100k queries per day 95th percentile 200ms 98 percent relevance", 3),
    ("Production GenAI pipeline fine-tuning Llama-2 LoRA company knowledge base RAG 50k daily queries hallucination detection guardrails", 3),
    ("Zero-trust security cloud-native OAuth2 mTLS Vault secrets management real-time threat detection 80 percent security incident reduction", 3),
]

TIER_KW = {
    3: ['production','deployed','million users','billion','scalable','distributed','microservices',
        'kubernetes','ci/cd','real-time','sla','team of','led','architected','reduced','percent'],
    2: ['api','database','authentication','docker','cloud','aws','gcp','azure','machine learning',
        'deep learning','neural network','nlp','redis','kafka','fine-tuned','bert','pipeline'],
    1: ['crud','rest','web app','mobile app','backend','frontend','react','django','flask',
        'sql','javascript','jwt','dashboard'],
    0: ['calculator','todo','hello world','game','basic','simple','tkinter','pygame','html css'],
}

def hand_feats(text):
    t = text.lower()
    hits = [sum(1 for kw in TIER_KW[i] if kw in t) for i in [0,1,2,3]]
    return [
        len(t.split())/100.0,
        hits[0]/5.0, hits[1]/8.0, hits[2]/10.0, hits[3]/12.0,
        float(bool(re.search(r'\d+%|\d+x|million|billion|\d+k users', t))),
        float(bool(re.search(r'deploy|production|cloud|k8s|docker', t))),
        float(bool(re.search(r'\d+[km]|million|billion|thousand', t))),
        float(bool(re.search(r'reduc|increas|improv|optim|achiev', t))),
        float(bool(re.search(r'team|led|manag|collaborat', t))),
        sum(1 for kws in TIER_KW.values() for kw in kws if kw in t) / 20.0,
    ]

np.random.seed(42)
aug = []
for text, label in RAW:
    aug.append((text, label))
    words = text.split()
    for _ in range(7):
        mask = np.random.random(len(words)) > np.random.uniform(0.05, 0.12)
        v = ' '.join(w for w, k in zip(words, mask) if k)
        if len(v.split()) > 5:
            aug.append((v, label))

df3 = pd.DataFrame(aug, columns=['description','label']).sample(frac=1, random_state=42).reset_index(drop=True)
y3 = df3['label'].values

print(f"  Dataset: {len(df3)} samples (60 seed x ~8 augmented)")
for i, n in enumerate(LABEL_NAMES3):
    print(f"    {n:14s}: {(y3==i).sum()} ({(y3==i).mean()*100:.1f}%)")

tfidf3 = TfidfVectorizer(max_features=3000, ngram_range=(1,2),
                          sublinear_tf=True, min_df=1, stop_words='english')
X3_tfidf = tfidf3.fit_transform(df3['description'])
X3_hand  = sp.csr_matrix(np.array([hand_feats(t) for t in df3['description']]))
X3 = sp.hstack([X3_tfidf, X3_hand])

X3_tr, X3_te, y3_tr, y3_te = train_test_split(
    X3, y3, test_size=0.2, random_state=42, stratify=y3
)

rf3 = RandomForestClassifier(n_estimators=300, max_depth=20,
                              min_samples_leaf=2, random_state=42, n_jobs=-1)
rf3.fit(X3_tr, y3_tr)
y3_pred  = rf3.predict(X3_te)
acc3  = accuracy_score(y3_te, y3_pred)
f1m3  = f1_score(y3_te, y3_pred, average='macro')
kap3  = cohen_kappa_score(y3_te, y3_pred)
cv3   = cross_val_score(rf3, X3, y3, cv=cv5, scoring='f1_macro', n_jobs=-1)

print(f"\n  Accuracy       : {acc3:.4f} ({acc3*100:.2f}%)")
print(f"  Macro F1       : {f1m3:.4f}")
print(f"  Cohen's Kappa  : {kap3:.4f}")
print(f"  CV F1 (5-fold) : {cv3.mean():.4f} +/- {cv3.std():.4f}")
print(f"\n{classification_report(y3_te, y3_pred, target_names=LABEL_NAMES3, digits=4)}")

print("  Real-world prediction test:")
examples = [
    ("Basic calculator app Python", "Beginner"),
    ("E-commerce site Django REST Stripe payment authentication", "Intermediate"),
    ("Fraud detection AWS SageMaker XGBoost 50k transactions per day", "Advanced"),
    ("Scaled fintech platform 5M users led team 8 reduced churn 25 percent production", "Production"),
]
for text, expected in examples:
    fv = sp.hstack([tfidf3.transform([text]), sp.csr_matrix([hand_feats(text)])])
    pred = LABEL_NAMES3[rf3.predict(fv)[0]]
    match = "OK" if pred == expected else "MISS"
    print(f"    [{match}] '{text[:55]}' -> {pred}")

cm3 = confusion_matrix(y3_te, y3_pred)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm3, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=LABEL_NAMES3, yticklabels=LABEL_NAMES3, ax=axes[0])
axes[0].set_title('Project Complexity - Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
cm3n = cm3.astype(float) / cm3.sum(axis=1, keepdims=True)
sns.heatmap(cm3n, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=LABEL_NAMES3, yticklabels=LABEL_NAMES3, ax=axes[1])
axes[1].set_title('Normalised Confusion Matrix', fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('plots/03_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

with open('../data/models/project_complexity.pkl', 'wb') as f:
    pickle.dump({
        'clf': rf3, 'tfidf': tfidf3, 'label_names': LABEL_NAMES3,
        'accuracy': float(acc3), 'f1_macro': float(f1m3),
        'kappa': float(kap3), 'cv_mean': float(cv3.mean()),
    }, f)
print("  Model saved -> data/models/project_complexity.pkl")
print(f"  RESULT: Accuracy={acc3*100:.2f}% | F1={f1m3:.4f} | Kappa={kap3:.4f}")


# ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  MODELS 2 & 3 TRAINED SUCCESSFULLY")
print("=" * 60)
print(f"  Model 2 - Hiring Recommender : Acc={acc2*100:.2f}%  ROC-AUC={roc2:.4f}  F1-W={f1w2:.4f}")
print(f"  Model 3 - Project Complexity : Acc={acc3*100:.2f}%  F1={f1m3:.4f}  Kappa={kap3:.4f}")
print("=" * 60)
