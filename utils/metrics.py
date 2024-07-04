from sklearn.metrics import make_scorer

def skill_extraction_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of skill extraction.

    Args:
    y_true (list of lists): True skills for each resume
    y_pred (list of lists): Predicted skills for each resume

    Returns:
    float: Accuracy score
    """
    total_correct = 0
    total_skills = 0
    for true_skills, pred_skills in zip(y_true, y_pred):
        true_set = set(true_skills)
        pred_set = set(skill for skill, _ in pred_skills)
        total_correct += len(true_set.intersection(pred_set))
        total_skills += len(true_set)
    return total_correct / total_skills if total_skills > 0 else 0

# Create a scorer that can be used with sklearn's cross-validation and GridSearchCV
skill_accuracy_scorer = make_scorer(skill_extraction_accuracy)