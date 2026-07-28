import argparse
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from gpt4all import Embed4All
from langchain.embeddings import HuggingFaceEmbeddings
from nomic import embed


def read_pdf_unstructured_elements(pdf_path):
    from unstructured.partition.pdf import partition_pdf
    elements = partition_pdf(
        filename=str(pdf_path),
        # Unstructured Helpers
        strategy="hi_res",
        infer_table_structure=True,
        model_name="yolox"
    )
    return elements

def read_pdf(pdf_path):
    loader = DirectoryLoader(pdf_path, glob="**/*.pdf", show_progress=True)
    books = loader.load()

def read_pdf_unstructured_lib(pdf_path):
    from langchain_community.document_loaders import UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(str(pdf_path), show_progress=True)
    return loader.load()

def split_doc(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    return all_splits

def store_basic_docs(docs):
    #Separately store basic1 docs with their basic chroma embedding to play around with more serious ones..
    collection = client.create_collection(name="LR_Disco_2_docs")
    for idx, doc in enumerate(docs):
        collection.add(
            documents = [doc.page_content],
            ids=[str(idx)]
        )

def hf_embed_and_store(client):
    #Retrieve stored docs and try alternative embeddings
    doc_collection = client.get_collection(name="LR_Disco_2_docs")
    _ids = doc_collection.get()['ids']
    _docs = doc_collection.get()['documents']
    #HuggingFaceEmbeddings embedding
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = [embedder.embed_documents(doc) for doc in _docs]
    #Store to nomic embedding specific collection
    hf_collection = client.create_collection(name="LR_Disco_2_hf")
    hf_collection.add(
        embeddings = embeddings,
        documents = _docs,
        ids=_ids
    )

def nomic_embed_and_store(client):
    #Retrieve stored docs and try alternative embeddings
    doc_collection = client.get_collection(name="LR_Disco_2_docs")
    _ids = doc_collection.get()['ids']
    _docs = doc_collection.get()['documents']
    #Nomic embedding
    embeddings = embed.text(_docs, inference_mode="local")['embeddings']
    #Store to nomic embedding specific collection
    nomic_collection = client.create_collection(name="LR_Disco_2_nomic")
    nomic_collection.add(
        embeddings = embeddings,
        documents = _docs,
        ids=_ids
    )

def embed4all_embed_and_store(client):
    #Retrieve stored docs and try alternative embeddings
    doc_collection = client.get_collection(name="LR_Disco_2_docs")
    _ids = doc_collection.get()['ids']
    _docs = doc_collection.get()['documents']
    #Embed4All embedding
    embedder = Embed4All(device='gpu')
    embedded_chunks = [embedder.embed(doc) for doc in _docs]
    #Store to nomic embedding specific collection
    embed4all_collection = client.create_collection(name="LR_Disco_2_embed4all")
    embed4all_collection.add(
        embeddings = embedded_chunks,
        documents = _docs,
        ids=_ids
    )

client = None


def main(pdf_file):
    global client

    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    data = read_pdf_unstructured_lib(pdf_path)
    docs = split_doc(data)
    client = chromadb.PersistentClient(path="data/db/")
    store_basic_docs(docs)
    nomic_embed_and_store(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a PDF file and store embeddings in Chroma.")
    parser.add_argument("pdf_file", help="Path to the PDF file to process")
    args = parser.parse_args()
    main(args.pdf_file)