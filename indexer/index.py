from dotenv import load_dotenv
import os
import cocoindex as coco
import functools
from vector_db.client import get_client


load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("API_KEY_QDRANT")

COLLECTION_NAME = "DOCS"


@coco.transform_flow()
def text_to_embedding(
    text: coco.DataSlice[str],
) -> coco.DataSlice[list[float]]:
    return text.transform(
        coco.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

qdrant_connection = coco.add_auth_entry(
    "Qdrant",
    coco.targets.QdrantConnection(
        grpc_url=url+":6334",
        api_key=api_key
    )
)


@coco.flow_def(name="DocsRAG")
def docs_flow(flow_builder: coco.FlowBuilder,data_scope: coco.DataScope): 
    print("flow")
    data_scope["documents"] = flow_builder.add_source(
        coco.sources.LocalFile(
            path="docs",
            included_patterns=["*.txt"]
        )
    )
    print("✅ SOURCE CREADO")
    
    collector = data_scope.add_collector()


    with data_scope["documents"].row() as doc:
        print("📄 DOC ENCONTRADO:", doc["filename"])

        doc["chunks"] = doc["content"].transform(
            coco.functions.SplitRecursively(),
            chunk_size=500,
            chunk_overlap=100
        )
        print("✂️ CHUNKS GENERADOS")

        with doc["chunks"].row() as chunk:
            print("🧩 CHUNK:", chunk["text"])
            chunk["embedding"] = text_to_embedding(chunk["text"])
            print("🧠 EMBEDDING OK")
            collector.collect(
                id= coco.GeneratedField.UUID,
                filename = doc["filename"],
                location = chunk["location"],
                text = chunk["text"],
                embedding = chunk["embedding"]
            )
            print("✅ COLLECT OK")
    print("🚀 EXPORT INIT")
    collector.export(
        "doc_embeddings",
        coco.targets.Qdrant(
            collection_name=COLLECTION_NAME,
            connection = qdrant_connection,
        
        ),
        primary_key_fields=["id"]
    )
    print("🚀 EXPORT CONFIGURED")
    
@functools.cache
def get_qdrant_client():
    return get_client()
    
def search(query:str) -> coco.QueryOutput:
    client = get_qdrant_client()
    
    query_embedding= text_to_embedding.eval(query)
    
    search_results = client.query_points(
        collection_name = COLLECTION_NAME,
        query =  query_embedding,
        using="embedding",
        limit = 10,
        with_payload = True,
        with_vectors = True
    )
    return coco.QueryOutput(
        results=[
            {
                "filename":result.payload["filename"],
                "text":result.payload["text"],
                "embedding":result.vector,
                "score":result.score
            }
            for result in search_results.points
            if result.payload is not None
        ],
        query_info=coco.QueryInfo(
            embedding=query_embedding,
            similarity_metric=coco.VectorSimilarityMetric.COSINE_SIMILARITY,
        ),
    )
    
def _main():
    while True:
        query= input("Enter your query: ")
        
        if query=="":
            break
        
        query_output = search(query)
        print("Search results:")
        for result in query_output.results:
            print(f"[{result['score']:.3f}] {result['filename']}")
            print(f"     {result['text']}")
            print("---")
        print()
        
if __name__ == "__main__":
    load_dotenv()
    
    _main()
            
    


