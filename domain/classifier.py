from abc import ABC, abstractmethod

class GenderClassifier(ABC):

    @abstractmethod
    def predict(self, text: str) -> str:
        pass
