import pickle
import numpy as np
import pytest


# Load the pipeline file once for tests or interactive use
with open("pipeline.pkl", 'rb') as file:
    classification_model = pickle.load(file)


def test_pipeline_predicts():
    """Sanity check: pipeline can predict and returns probability vector."""
    sample = np.array([[30, 50000]])
    pred = classification_model.predict(sample)
    probs = classification_model.predict_proba(sample)[0]
    assert pred.shape[0] == 1
    assert probs.shape[0] >= 2


if __name__ == "__main__":
    # Interactive usage preserved for manual runs
    age = int(input("Please enter the age of the customer: "))
    salary = int(input("Please enter the Salary of the customer: "))
    raw_input = np.array([[age, salary]])
    prediction = classification_model.predict(raw_input)
    probabilities = classification_model.predict_proba(raw_input)[0]
    print("***********Prediction results*********************")
    print("=======================================================")
    if prediction[0] == 1:
        print("Customer will buy the Product")
        print(f"Confidence Level: {probabilities[1]:.2%}")
    else:
        print("Customer will not buy the Product")
        print(f"Confidence Level: {probabilities[0]:.2%}")
