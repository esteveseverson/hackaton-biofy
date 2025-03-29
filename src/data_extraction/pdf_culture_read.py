import os
from glob import glob

from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma


class PDFProcessor:
    def __init__(self, GROQ_API_KEY):
        # Configuração do embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Configuração do cliente Groq
        self.groq_client = Groq(api_key=GROQ_API_KEY)

        # Configuração do text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Tamanho aumentado para melhorar a qualidade dos embeddings
            chunk_overlap=200  # Overlap aumentado para manter contexto
        )

    def process_pdf(self, path: str) -> None:
        """Processa um único PDF e armazena seus embeddings"""
        try:
            print(f"Processando arquivo: {path}")

            # Carrega o documento PDF
            loader = PyPDFLoader(file_path=path)
            documents = loader.load()

            # Divide o texto em chunks
            chunks = self.text_splitter.split_documents(documents)

            # Armazena no ChromaDB
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory='src/chroma_db'
            )

            print(f"PDF {path} processado com sucesso!")
        except Exception as e:
            print(f"Erro ao processar {path}: {str(e)}")

    def query_groq(self, question: str) -> str:
        """Envia uma consulta para a API da Groq"""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente especialista em documentos."},
                    {"role": "user", "content": question}
                ],
                model="deepseek-r1-distill-llama-70b",
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro na consulta à Groq: {str(e)}"


if __name__ == '__main__':
    # Cria o processador
    processor = PDFProcessor()

    # Processa todos os PDFs no diretório
    PDF_PATH = 'src/data_extraction/files/'
    all_pdfs = glob(os.path.join(PDF_PATH, '*.pdf'))

    if not all_pdfs:
        print(f"Nenhum PDF encontrado em {PDF_PATH}")
    else:
        for pdf in all_pdfs:
            processor.process_pdf(pdf)

    # Exemplo de consulta à Groq
    resposta = processor.query_groq("Me responda se vc consegue ler os arquivos enviados. Sobre oq um deles fala!? Meu querido!")
    print("\nResposta da Groq:")
    print(resposta)
