from http import HTTPStatus

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document

from src.config.config import Settings
from src.rag_chain import RAGQueryEngine
from src.schemas.input import InputMessage, OutputMessage

app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', status_code=HTTPStatus.OK)
def read_root():
    return {'message': 'API Hackaton'}


@app.post('/', response_model=OutputMessage, status_code=HTTPStatus.OK)
def ask_ia(message: InputMessage):
    def separar_resposta(texto):
        # Dividindo o texto com base nas tags <think> e </think>
        partes = texto.split('<think>')
        
        # Verificando se há uma parte de pensamento e uma resposta final
        if len(partes) > 1:
            pensamento = partes[1].split('</think>')[0].strip()  # Extrai o conteúdo de pensamento
            resposta_final = texto.split('</think>')[-1].strip()  # Extrai a resposta final após </think>
            return pensamento, resposta_final
        else:
            # Se não houver <think>, retorna o texto inteiro como resposta final
            return "", texto.strip()

    
    GROQ_API_KEY = Settings().GROQ_API_KEY
    
    # Diretório onde o ChromaDB está armazenado
    CHROMA_DIR = "src/chroma_db"
    
    # Inicializar o motor de consulta
    query_engine = RAGQueryEngine(
        groq_api_key=GROQ_API_KEY, 
        persist_directory=CHROMA_DIR
    )
    docs = query_engine.get_relevant_documents(message.message, k=6)

    return_list = []
    for i, doc in enumerate(docs):
        return_list.append({
            'doc': f'{i + 1}',
            'author': doc.metadata.get('author'),
            'total_pages': doc.metadata.get('total_pages'),
            'page': doc.metadata.get('page')
        })
    print(return_list)

    question = message.message
    
    # Opção 2: Obter resposta completa
    print("\n=== Resposta completa da consulta RAG ===")
    response = query_engine.answer_question(question)
    
    print(f"\nPergunta: {response['question']}")
    print(f"\nResposta: {response['answer']}")
    
    think, answer = separar_resposta(response['answer'])

    return OutputMessage(
        query=message.message,
        think=think,
        answer=answer,
        used_docs=return_list,
    )