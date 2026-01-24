from domain.recommender import RecommendationStrategy

class FaissRecommender(RecommendationStrategy):

    def __init__(self, embedder, indexer):
        self.embedder = embedder
        self.indexer = indexer
    
    def recommend(self, query: str, gender=None, top_k: int = 10):
        vector = self.embedder.embed(query)
        results = self.indexer.search(vector, top_k)

        if gender:
            results = [
                r for r in results
                if r.get("gender", "").lower() == gender.lower()
            ]

        return results

