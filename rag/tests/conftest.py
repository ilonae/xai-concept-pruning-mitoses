import pytest
from sentence_transformers import SentenceTransformer

from fixtures import EVAL_QUERIES, FIXTURE_DOCUMENTS  # noqa: F401 (re-exported for tests)

# Shared embedding model (loaded once)-

@pytest.fixture(scope="session")
def embed_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# In-memory ChromaDB index 
@pytest.fixture(scope="session")
def in_memory_index(embed_model):
    """
    Builds LlamaIndex VectorStoreIndex backed by in-memory ChromaDB collection, uses fixture documents
    """
    import chromadb
    from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    # In-memory ChromaDB
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("test_collection")

    # Configure LlamaIndex to use our embed model (no LLM needed for retrieval)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    Settings.llm = None 

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = [
        Document(
            text=doc["text"],
            metadata={
                "file_name": doc["source"],
                "page_label": doc["page"],
            },
            id_=doc["id"],
        )
        for doc in FIXTURE_DOCUMENTS
    ]

    index = VectorStoreIndex.from_documents(documents,storage_context=storage_context)
    return index


