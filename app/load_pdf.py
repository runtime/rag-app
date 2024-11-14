import argparse
import os
import logging
import psycopg2
from psycopg2.extras import execute_batch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings  # Example, adjust if you use a different embedding function
from langchain.document_loaders import PyPDFDirectoryLoader
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_PATH = "data"  # Folder for PDFs


def main():
    print("[main]")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents(FILE_PATH)
    chunks = split_documents(documents)
    add_to_postgres(chunks)


def load_documents(directory: str):
    """Load documents from a directory of PDFs."""
    print("[load_documents] Loading PDFs from:", directory)
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()


def split_documents(documents: list[Document]):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def get_embeddings():
    """Return embedding function (adjust as necessary)."""
    return OpenAIEmbeddings().embed_query  # Example function


def add_to_postgres(chunks: list[Document]):
    """Store chunks and their embeddings in PostgreSQL."""
    embeddings = get_embeddings()

    # Establish DB connection
    connection = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = connection.cursor()

    # Ensure pgvector extension is ready
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            chunk_id TEXT,
            content TEXT,
            embedding VECTOR(1536) -- Adjust dimension to match your embeddings
        );
    """)

    # Prepare batch insert
    insert_query = """
        INSERT INTO document_chunks (chunk_id, content, embedding)
        VALUES (%s, %s, %s)
    """
    records = []
    for chunk in chunks:
        chunk_id = f"{chunk.metadata.get('source')}:{chunk.metadata.get('page')}"
        content = chunk.page_content
        embedding = embeddings(content)
        records.append((chunk_id, content, embedding))

    execute_batch(cursor, insert_query, records)
    connection.commit()
    cursor.close()
    connection.close()

    print(f"Inserted {len(records)} chunks into PostgreSQL.")


def clear_database():
    """Clear the document_chunks table."""
    connection = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS document_chunks;")
    connection.commit()
    cursor.close()
    connection.close()
    print("Database cleared.")


if __name__ == "__main__":
    main()
