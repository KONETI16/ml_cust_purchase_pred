import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["preprocess"]

# Load Data
df = pd.read_csv("data/Purchase_dataset.csv")
X = df[['Age', 'Salary']]
y = df['Purchased']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y
)

# Fit preprocessing scale parameters
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save intermediate numpy arrays
np.save("data/X_train_scaled.npy", X_train_scaled)
np.save("data/X_test_scaled.npy", X_test_scaled)
np.save("data/y_train.npy", y_train.to_numpy())
np.save("data/y_test.npy", y_test.to_numpy())

# Instantiate the pipeline base structure and store it
base_pipeline = Pipeline([('scaler', scaler), ('classifier', None)])
# Save the preprocessing-only pipeline skeleton to a distinct file so
# the training stage can consume it and produce the final `customer_pipeline.pkl`.
with open("customer_pipeline_base.pkl", "wb") as f:
    pickle.dump(base_pipeline, f)
