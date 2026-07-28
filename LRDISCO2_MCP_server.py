from pathlib import Path

import chromadb
from gpt4all import Embed4All
from mcp.server.fastmcp import FastMCP


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = BASE_DIR / "data" / "db"
#Use Embed4All embedded collection
COLLECTION_NAME = "LR_Disco_2_embed4all"


# -----------------------------------------------------------------------------
# ChromaDB
# -----------------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# -----------------------------------------------------------------------------
# Embedding Model
# -----------------------------------------------------------------------------

embedder = Embed4All()


# -----------------------------------------------------------------------------
# MCP Server
# -----------------------------------------------------------------------------

mcp = FastMCP("Land Rover Disco 2 ChromaDB RAG")


@mcp.tool()
def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Search the ChromaDB knowledge base for documents
    relevant to the user's query.
    """

    # Generate an embedding for the user's query
    embedded_query = embedder.embed(query)

    # Search ChromaDB using the query embedding
    results = collection.query(query_embeddings=[embedded_query], n_results=n_results,)

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return "No relevant documents found."

    output = []

    for index, document in enumerate(documents):
        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        output.append(
            f"Result {index + 1}\n"
            f"Distance: {distance}\n\n"
            f"{document}"
        )

    return "\n\n---\n\n".join(output)


# -----------------------------------------------------------------------------
# Application Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()