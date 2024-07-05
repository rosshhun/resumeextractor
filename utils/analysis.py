import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from config import OUTPUT_DIR
import os

def analyze_feature_importance(skill_extractor):
    feature_importance = skill_extractor.get_feature_importance()
    feature_names = ['cosine_sim', 'fuzzy_sim', 'context_sim', 'ngram_sim', 'jaccard_sim', 'levenshtein_dist']

    if len(feature_importance) != len(feature_names):
        feature_names = [f'Feature {i + 1}' for i in range(len(feature_importance))]

    feature_dict = dict(zip(feature_names, feature_importance))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(feature_dict.values()), y=list(feature_dict.keys()))
    plt.title('Feature Importance in Skill Extraction')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    plt.close()

    return feature_dict

def error_analysis(y_true, y_pred, known_skills):
    y_true_flat = [skill for skills in y_true for skill in skills]
    y_pred_flat = [skill for skills in y_pred for skill, _ in skills]

    all_skills = set(y_true_flat + y_pred_flat)

    y_true_binary = [[1 if skill in skills else 0 for skill in all_skills] for skills in y_true]
    y_pred_binary = [[1 if skill in [s for s, _ in skills] else 0 for skill in all_skills] for skills in y_pred]

    cm = confusion_matrix(np.array(y_true_binary).ravel(), np.array(y_pred_binary).ravel())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Skill Extraction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    skill_metrics = {}
    for i, skill in enumerate(all_skills):
        true_positives = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t[i] == 1 and p[i] == 1)
        false_positives = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t[i] == 0 and p[i] == 1)
        false_negatives = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t[i] == 1 and p[i] == 0)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        skill_metrics[skill] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    sorted_skills = sorted(skill_metrics.items(), key=lambda x: x[1]['f1_score'], reverse=True)

    return {
        'confusion_matrix': cm,
        'skill_metrics': dict(sorted_skills)
    }