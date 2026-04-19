from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge"
FAISS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "faiss_index"

_embeddings_cache = None
_vector_store_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    return _embeddings_cache

def build_vector_store():
    loader = DirectoryLoader(
        str(KNOWLEDGE_DIR), glob="*.md", loader_cls=TextLoader
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, get_embeddings())
    db.save_local(str(FAISS_PATH))
    
    global _vector_store_cache
    _vector_store_cache = db
    return db

def load_vector_store():
    global _vector_store_cache
    if _vector_store_cache is None:
        _vector_store_cache = FAISS.load_local(
            str(FAISS_PATH), get_embeddings(), allow_dangerous_deserialization=True
        )
    return _vector_store_cache

if __name__ == "__main__":
    print("building vector store...")
    build_vector_store()
    print("done")
