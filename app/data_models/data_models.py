#define a data model for material estimation results

from pydantic import BaseModel
from typing import List, Optional

class MaterialEstimationResult(BaseModel):

    prediction: int
    probabilities: list



class MaterialEstimationRequest(BaseModel):

    autoID: str
    SeniorCity: str
    Partner: str
    Dependents: str
    Service1: str
    Service2: str
    Security: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    Charges: str
    Demand: str
    Class: str