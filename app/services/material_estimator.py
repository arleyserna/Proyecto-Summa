import joblib
import numpy as np
from catboost import CatBoostClassifier
from app.utils.config import MODEL_PATH


class MaterialEstimator:

    """Utility wrapper around a pre-trained model for material estimation.

    Design notes:
    - We lazy-load the model on first use to avoid import-time errors when the
      model file is not present during development.
    - `predict` and `predict_proba` are static methods that accept only the
      features array so calls like `MaterialEstimator.predict(X)` work.
    """

    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            try:
                cls._model = joblib.load(MODEL_PATH)
            except Exception as e:
                # Re-raise with a clearer message; this will only occur when the
                # caller actually requires the model (predict call).
                raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")
        return cls._model

    @staticmethod
    def predict(features: np.ndarray) -> np.ndarray:
        """Predict class labels for the given features.

        :param features: A 2D numpy array where each row represents an instance.
        :return: A 1D numpy array of predicted class labels.
        """
        model = MaterialEstimator._get_model()
        return model.predict(features)

    @staticmethod
    def predict_proba(features: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given features.

        :param features: A 2D numpy array where each row represents an instance.
        :return: A 2D numpy array of predicted class probabilities.
        """
        model = MaterialEstimator._get_model()
        return model.predict_proba(features).astype(float).tolist()