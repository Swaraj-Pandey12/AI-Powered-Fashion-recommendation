from abc import ABC, abstractmethod

class RecommendationStrategy(ABC):

    @abstractmethod
    def recommend(self, query: str, top_k: int = 5):
        pass
