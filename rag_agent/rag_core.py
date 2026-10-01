"""Piezas compartidas entre ingest.py, check_index.py y el agente: rutas, embeddings y ChromaDB.

Basado en el starter kit del Lab 6. Embeddings locales con multilingual-e5-small:
gratis, sin cuota y las transcripciones NO salen de la computadora al indexar.

Prueba tu instalación (descarga el modelo la primera vez, ~490 MB):
    python -m rag_agent.rag_core
"""

from pathlib import Path

from dotenv import load_dotenv

AGENT_DIR = Path(__file__).resolve().parent          # consulta-juntas/rag_agent
PROJECT_DIR = AGENT_DIR.parent                       # consulta-juntas
REPO_DIR = PROJECT_DIR.parent                        # Ajente_Auditori_IA
AUDITORIA_DIR = REPO_DIR / "auditoria-juntas"        # el proyecto de Arturo (NO se modifica)
TRANSCRIPTS_DIR = AUDITORIA_DIR / "transcripts"
AUDITORIAS_DIR = AUDITORIA_DIR / "auditorias"        # JSON de cada auditoría

load_dotenv(AGENT_DIR / ".env")  # la misma .env que usa `adk web`

DB_DIR = PROJECT_DIR / "chroma_db"  # ruta ABSOLUTA: adk web corre desde otro directorio
COLLECTION = "juntas"


def embedding_function():
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small",  # máx. 512 tokens (~300 palabras) por texto
        device="cpu",
        normalize_embeddings=True,
    )


def get_collection():
    """Colección persistente con distancia coseno."""
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_function(),
        configuration={"hnsw": {"space": "cosine"}},
    )


if __name__ == "__main__":
    vec = embedding_function()(["¿Funciona el modelo de embeddings?"])[0]
    print(f"✅ Embeddings listos: vectores de {len(vec)} dimensiones. Índice en {DB_DIR}")
    print(f"✅ Transcripciones en {TRANSCRIPTS_DIR} ({'existe' if TRANSCRIPTS_DIR.exists() else '❌ NO EXISTE'})")
