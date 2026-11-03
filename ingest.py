import truststore

truststore.inject_into_ssl()
import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

import config
from models import get_vector_store, setup_models


def reset_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]
    if config.COLLECTION_NAME in existing:
        client.delete_collection(config.COLLECTION_NAME)


def build_index():
    setup_models()
    reset_collection()

    documents = SimpleDirectoryReader(config.DATA_DIR, recursive=True).load_data()
    print(f"נטענו {len(documents)} מסמכים")

    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"נוצרו {len(nodes)} חלקים (chunks)")

    _, vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    print("האינדקס נשמר בתיקייה", config.CHROMA_DIR)


if __name__ == "__main__":
    build_index()
