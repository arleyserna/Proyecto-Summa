from adapters import client_adapter
from catboost import CatBoostClassifier
from services.material_estimator import MaterialEstimator
from utils.config import MODEL_PATH, PROJECT_ROOT
import pandas as pd

sample_client_data = {

    "autoID": "9695-TERGH",
    "SeniorCity": "0",
    "Partner": "No",
    "Dependents": "No",
    "Service1": "Yes",
    "Service2": "No",
    "Security": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "Charges": "96.05",
    "Demand": "",
    "Class": ""

}

adapter = client_adapter.ClientDataAdapter(sample_client_data)

df = adapter.convert_to_model_format()
print("Client Data Summary:\n")
print(df)
print(PROJECT_ROOT)


predictions = MaterialEstimator.predict(df.values)
predicted_probabilities = MaterialEstimator.predict_proba(df.values)
print(f"Predicted Classes: {predictions}")
print(f"Predicted Probabilities: {predicted_probabilities}")