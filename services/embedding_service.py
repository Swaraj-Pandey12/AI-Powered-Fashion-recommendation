class EmbeddingService:

    def __init__(self, model):
        self.model = model

    def embed(self, text: str):
        return self.model.encode([text])
