import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    """Loads data from a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)

def split_data(data, target_column):
    """Splits data into training and testing sets."""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred):
    """Evaluates the performance of a model."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, report

def main():
    """Main function for the data-automation-tool."""
    file_path = 'data.csv'
    target_column = 'target'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(data, target_column)
    # Train a model and make predictions
    # For demonstration purposes, we'll use a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, report = evaluate_model(y_test, y_pred)
    print('Model Accuracy:', accuracy)
    print('Classification Report:\n', report)

if __name__ == '__main__':
    main()