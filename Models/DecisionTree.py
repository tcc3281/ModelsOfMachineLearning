import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    @staticmethod
    def read_data_csv(file_path, target):
        data = pd.read_csv(file_path)
        X = data.drop(target, axis=1)
        y = data[target]
        encoder = OneHotEncoder()
        encoded_features = encoder.fit_transform(X)
        encoded_data = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out())
        encoded_data[target] = y.values
        return encoded_data

    @staticmethod
    def calculate_entropy(data, target):
        n = len(data)
        entropy = 0
        for i in data[target].unique():
            p = len(data[data[target] == i]) / n
            entropy += -p * np.log2(p)
        return entropy

    @staticmethod
    def calculate_information_gain(data, attribute, target):
        n = len(data)
        entropy = DecisionTree.calculate_entropy(data, target)
        gain = entropy
        for i in data[attribute].unique():
            p = len(data[data[attribute] == i]) / n
            gain -= p * DecisionTree.calculate_entropy(data[data[attribute] == i], target)
        return gain

    @staticmethod
    def train_model_decision_tree(data, target):
        X = data.drop(target, axis=1)
        y = data[target]
        clf = DecisionTreeClassifier()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf.fit(X_train, y_train)
        return clf

    @staticmethod
    def evaluate_model_decision_tree(model, data, target):
        X = data.drop(target, axis=1)
        y = data[target]
        cvs=cross_val_score(model, X, y, cv=5)
        F1=cvs.mean()
        confuse_matrix = pd.crosstab(y, model.predict(X), rownames=['Actual'], colnames=['Predicted'])
        res=[cvs,confuse_matrix,F1]
        return res
    @staticmethod
    def predict_decision_tree(model, data, target):
        X = data.drop(target, axis=1)
        return model.predict(X)