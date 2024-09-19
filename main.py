from Models.DecisionTree import DecisionTree as dt

if __name__ == '__main__':
    data = dt.read_data_csv('./data/car_evaluation.csv', 'class')
    target = 'class'
    clf = dt.predict_decision_tree(data, target)
    evaluation = dt.evaluate_model_decision_tree(clf, data, target)

