import pickle
import numpy as np

import warnings

#load the pipeline file

with open("pipeline.pkl", 'rb') as file:
    classification_model = pickle.load(file)

#get live user input
age = int(input("Please enter the age of the customer: "))
salary = int(input("Please enter the Salary of the customer: "))

raw_input = np.array([[age, salary]])

#predict using raw_input
#pipeline automatically applies the saved scaler operation on the inputs

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
