from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluation(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    evalu = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
    return evalu