import os
from glob import glob

from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import pandas as pd

def csv_processor(path: str) -> None:
    try:
        # Try to read the CSV file, skipping problematic lines and specifying the delimiter
        df = pd.read_csv(
            path,
            encoding='ISO-8859-1',
            delimiter=';'
        )
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return None

    text_data = df.astype(str).agg(' '.join, axis=1).tolist()

    embedding = HuggingFaceEmbeddings(model=...)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Process each string in the list text_data separately
    chunks = []
    for text in text_data:
        # Apply split_text on each string individually
        chunks.extend(text_splitter.split_text(text))  # `split_text` needs individual strings

    # Wrap each chunk of text as a Document with an ID
    documents = [Document(page_content=chunk, metadata={"id": idx}) for idx, chunk in enumerate(chunks)]

    vector_store = Chroma(
        persist_directory='src/chroma_db',
        embedding_function=embedding,
    )
    vector_store.add_documents(documents)  # Now passing documents with id and content

    print(f'CSV processed and stored in chroma_db: {path}')
    return None