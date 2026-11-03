import chromadb
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def setup_models():
    if not config.GOOGLE_API_KEY:
        raise RuntimeError("חסר GOOGLE_API_KEY בקובץ .env")

    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=config.EMBED_MODEL,
        api_key=config.GOOGLE_API_KEY,
    )
    Settings.llm = GoogleGenAI(
        model=config.LLM_MODEL,
        api_key=config.GOOGLE_API_KEY,
        temperature=0.1,
    )


def get_vector_store():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, ChromaVectorStore(chroma_collection=collection)
