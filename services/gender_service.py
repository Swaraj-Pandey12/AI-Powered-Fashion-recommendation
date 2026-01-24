import pickle
from domain.classifier import GenderClassifier

class GenderService(GenderClassifier):

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, text: str) -> str:
        return self.model.predict([text])[0]
