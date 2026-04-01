"""
RAG pipeline: index building, loading, and querying

LLM selection (checked in order):
  1. GROQ_API_KEY env var / st.secrets 
  2. fallback: Ollama (local)
"""

import os
import urllib.request
from pathlib import Path

import chromadb
import fitz  # pymupdf
from llama_index.core import (Document, Settings, StorageContext, VectorStoreIndex)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PAPERS_DIR = Path(__file__).parent / "papers"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "xai_papers"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama3-8b-8192"
OLLAMA_MODEL = "llama3.2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Papers to auto-download when not present locally (arXiv open access)
REMOTE_PAPERS = {
    "crp_achtibat_2022.pdf": "https://arxiv.org/pdf/2206.03208",
    "lrp_bach_2015.pdf": (
        "https://journals.plos.org/plosone/article/file"
        "?id=10.1371/journal.pone.0130140&type=printable"
    ),
}


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


def ensure_papers(papers_dir: Path = PAPERS_DIR) -> None:
    """
    Download remote papers into papers_dir if they are not already present, only fetches files listed in REMOTE_PAPERS
    """
    papers_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in REMOTE_PAPERS.items():
        dest = papers_dir / filename
        if not dest.exists():
            print(f"Downloading {filename}…")
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved to {dest}")


def configure_settings(
    groq_api_key: str | None = None,
    embed_model_name: str = EMBED_MODEL_NAME,
):
    """
    Configure LlamaIndex - Groq if a key is provided or found in the environment, otherwise fallback to local Ollama instance
    """
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    key = groq_api_key or os.getenv("GROQ_API_KEY")
    if key:
        from llama_index.llms.groq import Groq  # pip install llama-index-llms-groq
        llm = Groq(model=GROQ_MODEL, api_key=key)
    else:
        from llama_index.llms.ollama import Ollama
        llm = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)

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
