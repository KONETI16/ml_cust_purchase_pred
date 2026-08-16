import numpy as np
import yaml
import pickle
from sklearn.linear_model import LogisticRegression

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

# Load scaled data arrays
X_train_scaled = np.load("data/X_train_scaled.npy")
y_train = np.load("data/y_train.npy")

# Load intermediate pipeline shell containing the fitted scaler
with open("customer_pipeline_base.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Train the model component 
model = LogisticRegression(random_state=params["random_state"], solver=params["solver"])
model.fit(X_train_scaled, y_train)

# Inject trained model weights directly into the pipeline object steps
pipeline.steps[1] = ('classifier', model)

# Overwrite complete end-to-end production-ready pipeline
with open("customer_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
