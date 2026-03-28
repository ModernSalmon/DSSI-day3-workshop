"""
This module encapsulates model inference.
"""

import pandas as pd
from src.data_processor import preprocess
from src.model_registry import retrieve
from src.config import appconfig

def get_prediction(**kwargs):
    """
    Get prediction for given data.
        Parameters:
            kwargs: Keyworded argument list containing the data for prediction
        Returns:
            dict: Predicted class and malignant probability
    """
    clf, features = retrieve(appconfig['Model']['name'])
    pred_df = pd.DataFrame(kwargs, index=[0])
    pred_df = preprocess(pred_df)
    pred = clf.predict(pred_df[features])
    prob = clf.predict_proba(pred_df[features])
    malignant_probability = float(prob[0][1])
    return {
        "prediction": int(pred[0]),
        "malignant_probability": round(malignant_probability, 3),
    }
