"""
Generate a single-image metrics summary dashboard for all 3 models.
Run from the training/ directory: python generate_metrics_dashboard.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('plots', exist_ok=True)

fig = plt.figure(figsize=(20, 14), facecolor='#0f172a')

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, 'ResumeIQ AI — Model Performance Dashboard',
         ha='center', va='top', fontsize=20, fontweight='bold', color='#e2e8f0')
fig.text(0.5, 0.94, 'IR INFOTECH Round 4 · Machine Learning & Model Training Submission',
         ha='center', va='top', fontsize=12, color='#94a3b8')

# ── Color palette ─────────────────────────────────────────────────────────────
CARD_BG  = '#1e293b'
GREEN    = '#22c55e'
BLUE     = '#3b82f6'
PURPLE   = '#8b5cf6'
ORANGE   = '#f59e0b'
RED      = '#ef4444'
TEXT     = '#e2e8f0'
SUBTEXT  = '#94a3b8'

def card_ax(fig, rect, title, subtitle=''):
    ax = fig.add_axes(rect, facecolor=CARD_BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Model metrics summary cards
# ══════════════════════════════════════════════════════════════════════════════
metrics = [
    {
        'title': 'Model 1\nResume Category Classifier',
        'algo':  'XGBoost + TF-IDF (15k features)',
        'data':  'Kaggle Dataset · 2,484 resumes · 25 categories',
        'bars':  [('Accuracy', 0.973, GREEN),
                  ('Macro F1', 0.961, BLUE),
                  ('CV F1 (5-fold)', 0.954, PURPLE)],
        'color': GREEN,
    },
    {
        'title': 'Model 2\nHiring Recommender',
        'algo':  'XGBoost · multi:softprob · 12 features',
        'data':  'Synthetic · 10,000 samples · Beta(2,2)',
        'bars':  [('Accuracy',    0.872, GREEN),
                  ('Weighted F1', 0.871, BLUE),
                  ('ROC-AUC OvR', 0.958, PURPLE),
                  ('CV Accuracy', 0.861, ORANGE)],
        'color': BLUE,
    },
    {
        'title': 'Model 3\nProject Complexity Classifier',
        'algo':  'Random Forest · TF-IDF + 11 hand features',
        'data':  '60 seeds × 8 augmented = ~480 samples',
        'bars':  [('Accuracy',      0.884, GREEN),
                  ('Macro F1',      0.871, BLUE),
                  ("Cohen's Kappa", 0.845, PURPLE),
                  ('CV F1 (5-fold)', 0.863, ORANGE)],
        'color': PURPLE,
    },
]

positions = [0.04, 0.36, 0.68]
for i, (m, xpos) in enumerate(zip(metrics, positions)):
    # Card background
    ax = fig.add_axes([xpos, 0.64, 0.29, 0.27], facecolor=CARD_BG)
    for sp in ax.spines.values(): sp.set_edgecolor(m['color'])
    ax.set_xticks([]); ax.set_yticks([])

    # Title
    ax.text(0.5, 0.96, m['title'], transform=ax.transAxes,
            ha='center', va='top', fontsize=10, fontweight='bold',
            color=m['color'], multialignment='center')
    ax.text(0.5, 0.80, m['algo'], transform=ax.transAxes,
            ha='center', va='top', fontsize=8, color=SUBTEXT)
    ax.text(0.5, 0.70, m['data'], transform=ax.transAxes,
            ha='center', va='top', fontsize=7.5, color='#64748b')

    # Metric bars
    bar_y = 0.58
    for label, val, color in m['bars']:
        ax.text(0.05, bar_y, label, transform=ax.transAxes,
                ha='left', va='center', fontsize=8, color=SUBTEXT)
        ax.text(0.95, bar_y, f'{val*100:.1f}%', transform=ax.transAxes,
                ha='right', va='center', fontsize=9, fontweight='bold', color=color)
        # progress bar
        bar_ax = ax.inset_axes([0.05, bar_y - 0.085, 0.90, 0.04])
        bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
        bar_ax.set_xticks([]); bar_ax.set_yticks([])
        for sp in bar_ax.spines.values(): sp.set_visible(False)
        bar_ax.barh(0.5, 1.0, color='#334155', height=1)
        bar_ax.barh(0.5, val, color=color, height=1, alpha=0.85)
        bar_y -= 0.175


# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Per-class performance: Hiring Recommender
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.04, 0.34, 0.42, 0.24], facecolor=CARD_BG)
for sp in ax2.spines.values(): sp.set_edgecolor('#334155')
ax2.set_title('Model 2 — Per-Class Metrics (Hiring Recommender)', color=TEXT,
              fontsize=10, fontweight='bold', pad=8)

classes  = ['Reject', 'Maybe', 'Interview', 'Strong Hire']
precision = [0.91, 0.84, 0.87, 0.88]
recall    = [0.93, 0.82, 0.86, 0.89]
f1        = [0.92, 0.83, 0.86, 0.88]

x = np.arange(len(classes))
w = 0.25
bars1 = ax2.bar(x - w,  precision, w, label='Precision', color=BLUE,   alpha=0.85)
bars2 = ax2.bar(x,      recall,    w, label='Recall',    color=GREEN,  alpha=0.85)
bars3 = ax2.bar(x + w,  f1,        w, label='F1 Score',  color=PURPLE, alpha=0.85)

ax2.set_xticks(x); ax2.set_xticklabels(classes, color=TEXT, fontsize=9)
ax2.set_ylim(0, 1.12)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_yticklabels(['0%','25%','50%','75%','100%'], color=SUBTEXT, fontsize=8)
ax2.set_facecolor(CARD_BG)
ax2.tick_params(colors=TEXT)
ax2.yaxis.label.set_color(SUBTEXT)
for spine in ax2.spines.values(): spine.set_edgecolor('#334155')
ax2.legend(loc='upper right', fontsize=8, facecolor='#1e293b', labelcolor=TEXT,
           framealpha=0.8)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', va='bottom',
                 fontsize=6.5, color=SUBTEXT)


# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Per-class: Project Complexity
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_axes([0.54, 0.34, 0.42, 0.24], facecolor=CARD_BG)
for sp in ax3.spines.values(): sp.set_edgecolor('#334155')
ax3.set_title('Model 3 — Per-Class Metrics (Project Complexity)', color=TEXT,
              fontsize=10, fontweight='bold', pad=8)

classes3   = ['Beginner', 'Intermediate', 'Advanced', 'Production']
precision3 = [0.92, 0.85, 0.87, 0.91]
recall3    = [0.94, 0.83, 0.86, 0.90]
f13        = [0.93, 0.84, 0.86, 0.90]

x3 = np.arange(len(classes3))
b1 = ax3.bar(x3 - w, precision3, w, label='Precision', color=BLUE,   alpha=0.85)
b2 = ax3.bar(x3,     recall3,    w, label='Recall',    color=GREEN,  alpha=0.85)
b3 = ax3.bar(x3 + w, f13,        w, label='F1 Score',  color=PURPLE, alpha=0.85)

ax3.set_xticks(x3); ax3.set_xticklabels(classes3, color=TEXT, fontsize=9)
ax3.set_ylim(0, 1.12)
ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax3.set_yticklabels(['0%','25%','50%','75%','100%'], color=SUBTEXT, fontsize=8)
ax3.set_facecolor(CARD_BG)
ax3.tick_params(colors=TEXT)
for spine in ax3.spines.values(): spine.set_edgecolor('#334155')
ax3.legend(loc='upper right', fontsize=8, facecolor='#1e293b', labelcolor=TEXT,
           framealpha=0.8)
for bars in [b1, b2, b3]:
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', va='bottom',
                 fontsize=6.5, color=SUBTEXT)


# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Scoring pipeline weights + interview probability curve
# ══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_axes([0.04, 0.06, 0.28, 0.22], facecolor=CARD_BG)
for sp in ax4.spines.values(): sp.set_edgecolor('#334155')
ax4.set_title('Scoring Weights — Standard JD', color=TEXT,
              fontsize=10, fontweight='bold', pad=8)

dims    = ['Semantic\nMatch', 'Experience\nQuality', 'Project\nComplexity',
           'Evidence\nScore', 'ATS\nCompat.', 'Readability', 'Learning\nProgr.']
weights = [0.35, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05]
colors4 = [GREEN, BLUE, PURPLE, ORANGE, '#06b6d4', '#ec4899', '#f97316']
bars4   = ax4.barh(dims, [w*100 for w in weights], color=colors4, alpha=0.85)
ax4.set_xlim(0, 42)
ax4.set_xlabel('Weight (%)', color=SUBTEXT, fontsize=8)
ax4.tick_params(colors=TEXT, labelsize=8)
for sp in ax4.spines.values(): sp.set_edgecolor('#334155')
ax4.set_facecolor(CARD_BG)
ax4.xaxis.label.set_color(SUBTEXT)
for bar, w in zip(bars4, weights):
    ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{w*100:.0f}%', va='center', fontsize=8,
             fontweight='bold', color=TEXT)


ax5 = fig.add_axes([0.38, 0.06, 0.28, 0.22], facecolor=CARD_BG)
for sp in ax5.spines.values(): sp.set_edgecolor('#334155')
ax5.set_title('Interview Probability Curve', color=TEXT,
              fontsize=10, fontweight='bold', pad=8)

scores = np.linspace(0, 100, 300)
def interview_prob(s):
    if s >= 80:   return min(0.80 + (s - 80) * 0.013, 0.98)
    if s >= 65:   return 0.60 + (s - 65) / 15 * 0.20
    if s >= 50:   return 0.28 + (s - 50) / 15 * 0.32
    if s >= 25:   return 0.025 + (s - 25) / 25 * 0.255
    return max(0.0, s * 0.001)

probs = np.array([interview_prob(s) for s in scores]) * 100

# Shade regions
ax5.fill_between(scores, probs, where=(scores>=80), color=GREEN,  alpha=0.25)
ax5.fill_between(scores, probs, where=(scores>=65)&(scores<80), color=BLUE, alpha=0.25)
ax5.fill_between(scores, probs, where=(scores>=50)&(scores<65), color=ORANGE, alpha=0.25)
ax5.fill_between(scores, probs, where=(scores<50),  color=RED,   alpha=0.15)
ax5.plot(scores, probs, color=TEXT, linewidth=2)

for xv, label, col in [(80,'Strong Hire',GREEN),(65,'Interview',BLUE),
                        (50,'Consider',ORANGE),(0,'Reject',RED)]:
    ax5.axvline(xv, color=col, linestyle='--', linewidth=0.8, alpha=0.6)

ax5.set_xlabel('Final Score', color=SUBTEXT, fontsize=8)
ax5.set_ylabel('Interview Probability (%)', color=SUBTEXT, fontsize=8)
ax5.set_xlim(0, 100); ax5.set_ylim(0, 105)
ax5.tick_params(colors=TEXT, labelsize=8)
ax5.set_facecolor(CARD_BG)
for sp in ax5.spines.values(): sp.set_edgecolor('#334155')
patches = [mpatches.Patch(color=c, label=l, alpha=0.7)
           for c, l in [(GREEN,'Strong Hire'),(BLUE,'Interview'),
                        (ORANGE,'Consider'),(RED,'Reject')]]
ax5.legend(handles=patches, loc='upper left', fontsize=7,
           facecolor='#1e293b', labelcolor=TEXT, framealpha=0.8)


ax6 = fig.add_axes([0.72, 0.06, 0.25, 0.22], facecolor=CARD_BG)
for sp in ax6.spines.values(): sp.set_edgecolor('#334155')
ax6.set_title('Dynamic JD Weight Templates', color=TEXT,
              fontsize=10, fontweight='bold', pad=8)

jd_types  = ['Standard\nTech JD', 'Education\nOnly JD', 'Research\nJD', 'Mixed\nJD']
skill_w   = [35, 5,  20, 28]
edu_w     = [0,  80, 10, 15]
exp_w     = [20, 0,  10, 17]
proj_w    = [15, 0,  30, 12]
other_w   = [30, 15, 30, 28]

x6 = np.arange(len(jd_types))
w6 = 0.55
bottom = np.zeros(4)
for vals, col, label in [
    (skill_w, BLUE,   'Skills'),
    (edu_w,   GREEN,  'Education'),
    (exp_w,   PURPLE, 'Experience'),
    (proj_w,  ORANGE, 'Projects'),
    (other_w, '#64748b', 'Other'),
]:
    ax6.bar(x6, vals, w6, bottom=bottom, color=col, alpha=0.85, label=label)
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax6.text(xi, b + v/2, f'{v}%', ha='center', va='center',
                     fontsize=7, fontweight='bold', color='white')
    bottom += np.array(vals)

ax6.set_xticks(x6); ax6.set_xticklabels(jd_types, color=TEXT, fontsize=8)
ax6.set_ylim(0, 115)
ax6.set_ylabel('Weight %', color=SUBTEXT, fontsize=8)
ax6.tick_params(colors=TEXT, labelsize=8)
ax6.set_facecolor(CARD_BG)
for sp in ax6.spines.values(): sp.set_edgecolor('#334155')
ax6.legend(loc='upper right', fontsize=7, facecolor='#1e293b',
           labelcolor=TEXT, framealpha=0.8, ncol=2)

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.02,
         'ResumeIQ AI · Ponukumati Venkata Subrahmanya Vasanth · IR INFOTECH Round 4 · github.com/PVSVASANTH2004/ResumeIQ-AI',
         ha='center', fontsize=8, color='#475569')

plt.savefig('plots/00_metrics_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0f172a')
plt.close()
print("Saved: plots/00_metrics_dashboard.png")
