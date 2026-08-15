#Testing the api 
import numpy as np
import requests

url = "http://127.0.0.1:8000/"

test_customers = [ {"age": 30, "salary": 50000},
                   {"age": 45, "salary": 80000},
                   {"age": 22, "salary": 30000}, 
                   {"age": 35, "salary": 60000},
                   {"age": 50, "salary": 100000}]

#test api endpoint health check
response = requests.get(url + "health")

try:
    if response.status_code == 200:
        print("API Health Check: Passed")
    else:
        print("API Health Check: Failed")
except Exception as e:
    print("Error during API Health Check:", str(e))


for i, customer in enumerate(test_customers, start=1):
    age = customer["age"]
    salary = customer["salary"]
    
    #test api endpoint prediction
    response = requests.get(url + "predict", params={"age": age, "salary": salary})
    
    try:
        if response.status_code == 200:
            result = response.json()
            print(f"Test Case {i}: Age={age}, Salary={salary} => Prediction: {result['prediction']}, Confidence Level: {result['confidence_level']}")
        else:
            print(f"Test Case {i}: Age={age}, Salary={salary} => API call failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error during API call for Test Case {i}: Age={age}, Salary={salary} => {str(e)}")
