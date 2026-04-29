import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    """
    Main function for the data-automation-tool. This tool is designed to 
    automate data preprocessing and simple machine learning tasks.

    Parameters:
    None

    Returns:
    None
    """
    # Load the dataset
    try:
        data = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("The file 'data.csv' was not found.")
        return

    # Preprocess the data
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest classifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    # Print the results
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()