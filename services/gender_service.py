import os
import pickle
from domain.classifier import GenderClassifier


class GenderService(GenderClassifier):

    def __init__(self, model_path: str):
        # check model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gender model not found: {model_path}")

        # load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, text: str) -> str:
        # handle empty input
        if not text:
            return "Unknown"

        try:
            prediction = self.model.predict([text])
            return prediction[0]

        except Exception as e:
            print("Gender prediction error:", e)
            return "Unknown"
