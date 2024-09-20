from Models.DecisionTree import DecisionTree as dt

if __name__ == '__main__':
    data = dt.read_data_csv('./data/car_evaluation.csv', 'class')
    target = 'class'
    clf=dt.train_model_decision_tree(data, target)
    evaluation = dt.evaluate_model_decision_tree(clf, data, target)
    print(f'F1: {evaluation[2]}')
    print(f'Cross Validation Score: {evaluation[0]}')
    print(f'Confusion Matrix:\n{evaluation[1]}')