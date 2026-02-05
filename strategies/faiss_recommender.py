from domain.recommender import RecommendationStrategy

class FaissRecommender(RecommendationStrategy):

    def __init__(self, embedder, indexer):
        self.embedder = embedder
        self.indexer = indexer

    def recommend(self, query: str, gender=None, top_k: int = 10):
        vector = self.embedder.embed(query)
        all_results = self.indexer.search(vector, top_k*3)  # get more to filter

        if gender:
            # filter by predicted gender
            filtered = [r for r in all_results if r.get("gender") == gender]
            return filtered[:top_k] if filtered else all_results[:top_k]

        return all_results[:top_k]
