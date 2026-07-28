# Land Rover Discovery II Maintenance Assistant (MCP Server)

This is a transformation of https://github.com/TommiA/LRDISCO2_RAG_LLAMA3 to MCP server.

This is an intersectional combination of Artificial Intelligence (AI) and specifically Large Language Models (LLM), Natural Language Processing, PDFs, Land Rovers and car maintenance - a rainy day summer project for fun.

This project implements a Retrieval-Augmented Generation (RAG) system with a quantized version of the LLAMA 3 8B Large Language Model. The system uses the Chroma vector database to store parts of the Land Rover Discovery II maintenance manual (not included in this repository) and provides a knowledge base about the Discovery II and its maintenance. Data is not included: See [read_pdf_to_chroma_langchain.py](read_pdf_to_chroma_langchain.py) to preprocesses the input PDF file to the data/pdf folder, dividing it to chunks of text and embedding, finally storing it to the Chroma DB

The core of this system is a Model Control Protocol (MCP) server that exposes a single tool for querying the knowledge base. This server architecture allows for flexible integration with various clients and applications.

## MCP Server Architecture

The system is built around a FastMCP server that provides a single tool for querying the knowledge base:

- **MCP Server**: `LRDISCO2_MCP_server.py` - A FastMCP server that exposes a single tool for querying the ChromaDB knowledge base
- **Knowledge Base**: ChromaDB with embeddings from the Land Rover Discovery II maintenance manual
- **Embedding Model**: GPT4All's Embed4All model for creating document embeddings
- **Query Tool**: `search_knowledge_base` - A single tool that searches the ChromaDB knowledge base for documents relevant to the user's query

The MCP server architecture allows for seamless integration with various clients and applications, making it easy to query the Land Rover Discovery II maintenance knowledge base from different environments.

## Core Components

### 1. Knowledge Base
The ChromaDB database stores chunks of text from the Land Rover Discovery II maintenance manual, each with an embedding vector created using the GPT4All Embed4All model. The database is stored at `data/db/` and uses the collection name `LR_Disco_2_embed4all`.

### 2. MCP Server
The `LRDISCO2_MCP_server.py` file implements the FastMCP server that exposes a single tool for querying the knowledge base. The server uses the following components:
- ChromaDB client connected to the persistent database
- Embed4All model for creating query embeddings
- FastMCP framework for handling MCP protocol

### 3. Query Tool
The `search_knowledge_base` function is the only tool exposed by the MCP server. It takes a user query and returns relevant documents from the ChromaDB knowledge base. The function:
- Creates an embedding for the user's query
- Searches ChromaDB using the query embedding
- Returns the top results with their distances and documents

## Usage

### Starting the MCP Server
To start the MCP server, run:
```bash
python LRDISCO2_MCP_server.py
```

The server will start listening for MCP requests and expose the `search_knowledge_base` tool.

### Using the MCP Server on LM Studio
See 'mcp.json' for adding this MCP server to your LM Studio instance, edit the path accordingly.

### Testing using Langchain client, local MCP Server and LM Studio
Run 'langchain_test_client.py', set LM Studio ip-addresses as needed in script or setting LM_STUDIO_BASEURL using .env, use OpenAI URL style. 

#### 'langchain_test_client.py' switches
The script supports the following command-line options:
* `-p`, `--prompt` : custom user prompt text
* `-i`, `--interactive` : start an interactive chat loop
