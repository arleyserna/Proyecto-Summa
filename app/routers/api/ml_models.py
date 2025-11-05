from fastapi import APIRouter, Depends
from app.services.material_estimator import MaterialEstimator
from app.data_models.data_models import MaterialEstimationRequest, MaterialEstimationResult
from app.adapters.client_adapter import ClientDataAdapter


router = APIRouter(
    prefix="/api/ml_models_api",
)

@router.get("/", summary="Get Materials Classifier API Status")
async def get_status():
    """
    Get the status of the Materials Classifier API.

    :return: A dictionary indicating the API is running.
    """
    return {"status": "Materials Classifier API is running"}



@router.post("/predict", tags=["Materials Classifier"], summary="Predict Alpha and Betha values", )
async def predict_materials(data: MaterialEstimationRequest) -> MaterialEstimationResult:
    """
    Predict Alpha and Betha values based on input data.

    :param data: Input data for prediction.
    :return: Predicted Alpha and Betha values.
    """

    adapter = ClientDataAdapter(data.model_dump())
    df = adapter.convert_to_model_format()
    predictions = MaterialEstimator.predict(df.values)
    proba = MaterialEstimator.predict_proba(df.values)
    return MaterialEstimationResult(prediction=predictions, probabilities=proba)

@router.post("/demand_forecast", tags=['Demand Forecast'], summary="Predict Demand values", )
async def predict_demand(months: int) -> str:
    """
    Predict Demand values based on input data.

    :param data: Input data for prediction.
    :return: Predicted Demand values.
    
    TODO: Implement demand prediction logic base on the Prophet or SARIMA models.
    
    """

    return [{"message": "Demand prediction not yet implemented."}]
