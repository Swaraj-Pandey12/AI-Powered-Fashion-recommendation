from domain.recommender import RecommendationStrategy

class FaissRecommender(RecommendationStrategy):

    def __init__(self, embedder, indexer):
        self.embedder = embedder
        self.indexer = indexer

#     def recommend(self, query: str, gender=None, top_k: int = 10):
#         vector = self.embedder.embed(query)
#         all_results = self.indexer.search(vector, top_k*3)  # get more to filter

#         if gender:
#             # filter by predicted gender
#             filtered = [r for r in all_results if r.get("gender") == gender]
#             return filtered[:top_k] if filtered else all_results[:top_k]

#         return all_results[:top_k]


    def recommend(self, query: str, gender=None, top_k: int = 20):
        vector = self.embedder.embed(query)
        all_results = self.indexer.search(vector, top_k * 3)

    # If gender is provided, use it only for ranking, not filtering
        men, women, others = [], [], []

        for r in all_results:
            if r.get("gender") == "Men":
                men.append(r)
            elif r.get("gender") == "Women":
                women.append(r)
            else:
                others.append(r)

    # Neutral or fashion queries → show both
        if gender is None:
            return (women + men + others)[:top_k]

    # Balanced output (NOT filter)
        balanced = []

    # Prefer predicted gender slightly
        if gender == "Women":
            balanced.extend(women[:top_k//2])
            balanced.extend(men[:top_k//2])
        else:
            balanced.extend(men[:top_k//2])
            balanced.extend(women[:top_k//2])

        balanced.extend(others)

        return balanced[:top_k]
