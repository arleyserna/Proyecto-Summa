from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.api.ml_models import router
import uvicorn

app = FastAPI(
    title="Alpha Betha Predictor API",
    description="API for ML Models: pre-trained CatBoost model for Alpha-Betha Classification + SARIMA, PROPHET for forecast demand.",
    version="1.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=2500,
        reload=True,
        log_level="info",
    )


