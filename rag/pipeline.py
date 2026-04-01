"""
RAG pipeline: index building, loading, and querying
"""

from pathlib import Path

import chromadb
import fitz  # pymupdf
from llama_index.core import (Document, Settings, StorageContext, VectorStoreIndex)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

PAPERS_DIR = Path(__file__).parent / "papers"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "xai_papers"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "llama3.2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def load_pdfs(papers_dir: Path) -> list[Document]:
    """
    Load PDFs using PyMuPDF (fitz) instead of pypdf, as it correctly decodes font encodings in LaTeX-compiled PDFs,
    where pypdf often returns binary garbage
    """
    documents = []
    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {papers_dir}.")

    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue  # image-only pages
            documents.append(Document(
                text=text,
                metadata={
                    "file_name": pdf_path.name,
                    "page_label": str(page_num + 1),
                },
            ))
        doc.close()

    return documents


def configure_settings(embed_model_name: str = EMBED_MODEL_NAME, ollama_model: str = OLLAMA_MODEL):
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    llm = Ollama(model=ollama_model, request_timeout=120.0)
    Settings.embed_model = embed_model
    Settings.llm = llm
    return embed_model


def build_or_load_index(
    papers_dir: Path = PAPERS_DIR, chroma_dir: Path = CHROMA_DIR, 
    collection_name: str = COLLECTION_NAME, force_rebuild: bool = False ):
    """
    Load the existing ChromaDB index or build from scratch if empty, returns (index, chroma_collection)
    """
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() == 0 or force_rebuild:
        if force_rebuild and chroma_collection.count() > 0:
            chroma_client.delete_collection(collection_name)
            chroma_collection = chroma_client.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        documents = load_pdfs(papers_dir)

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(documents)

        index = VectorStoreIndex(nodes, storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    return index, chroma_collection


def query(index: VectorStoreIndex, question: str, top_k: int = 5):
    """Run a RAG query, returns a LlamaIndex Response obj"""
    engine = index.as_query_engine(similarity_top_k=top_k)
    return engine.query(question)
