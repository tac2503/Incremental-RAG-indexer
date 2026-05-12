from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY_QDRANT")
url = os.getenv("QDRANT_URL")

COLLECTION = "DOCS"

def get_client():
    
    qdrant_client = QdrantClient(
        url=url,
        api_key=api_key,
        cloud_inference =True
    )
    return qdrant_client

def ensure_collection(vector_size=384):
    
    client = get_client()
    
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    
    
    if COLLECTION not in names:
        print("creando colección")
        client.create_collection(
            collection_name = COLLECTION,
            vectors_config ={
                "embedding": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            }
        )       

if __name__ == "__main__":
    ensure_collection()