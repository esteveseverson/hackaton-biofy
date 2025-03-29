from typing import Dict

from pydantic import BaseModel
from langchain.schema import Document


class InputMessage(BaseModel):
    message: str


class DocumentInfo(BaseModel):
    author: str
    total_pages: int
    page: int

# Modelo principal para a mensagem de saída
class OutputMessage(BaseModel):
    query: str
    think: str
    answer: str
    used_docs: list