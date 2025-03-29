from glob import glob

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def pdf_processor(path: str) -> None:
    loader = PyPDFLoader(file_path=path)
    documents = loader.load()
    embedding = OllamaEmbeddings(model='deepseek-r1:7b')

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_spliter.split_documents(documents=documents)

    vector_store = Chroma(
        persist_directory='src/chroma_db',
        embedding_function=embedding,
    )
    vector_store.add_documents(chunks)

    print('PDF processado e armazenado no chroma_db')
    return None


if __name__ == '__main__':
    PDF_PATH = 'src/data_extraction/files/'
    all_pdfs = glob(PDF_PATH + '*.pdf')

    for pdf in all_pdfs:
        pdf_processor(pdf)        
