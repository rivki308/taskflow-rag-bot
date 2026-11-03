import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "org_knowledge"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

TOP_K = 4
SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.70"))

NO_INFO_ANSWER = "אין לי מידע על כך במסמכי המערכת."
