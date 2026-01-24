# from sentence_transformers import SentenceTransformer

# from services.embedding_service import EmbeddingService
# from services.faiss_indexer import FaissIndexer
# from services.gender_service import GenderService
# from strategies.faiss_recommender import FaissRecommender
# from config import settings

# def bootstrap():

#     embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#     embedder = EmbeddingService(embedding_model)

#     indexer = FaissIndexer(
#         settings.FAISS_INDEX_PATH,
#         settings.METADATA_PATH
#     )

#     gender_service = GenderService(settings.GENDER_MODEL_PATH)

#     recommender = FaissRecommender(embedder, indexer)

#     return recommender, gender_service


from sentence_transformers import SentenceTransformer

from services.embedding_service import EmbeddingService
from services.faiss_indexer import FaissIndexer
from services.gender_service import GenderService
from strategies.faiss_recommender import FaissRecommender
from config import settings

def bootstrap():

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = EmbeddingService(embedding_model)

    indexer = FaissIndexer(
        settings.FAISS_INDEX_PATH,
        settings.METADATA_PATH
    )


    recommender = FaissRecommender(embedder, indexer)

    return recommender
