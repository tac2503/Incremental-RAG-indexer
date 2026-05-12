from vector_db.client import get_client
from qdrant_client.models import Filter, FilterSelector

COLLECTION ="DOCS"

def wipe_points():
    client=get_client()
    
    client.delete(
        collection_name = COLLECTION,
        points_selector=FilterSelector(
            filter=Filter()
        )
    )
    print(f"Todos los puntos eliminados de {COLLECTION}")

if __name__ == "__main__":
    wipe_points()