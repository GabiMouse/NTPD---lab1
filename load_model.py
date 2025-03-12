from joblib import load
import numpy as np

model_version = "model_v1.joblib"
model = load(model_version)

sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(sample_input)
print(f"Predykcja dla {sample_input}: {prediction}")

