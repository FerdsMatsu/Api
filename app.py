# app.py
from fastapi import FastAPI, Query
from kdtree_wrapper import lib, Tarv, TReg
from ctypes import POINTER,c_char
from pydantic import BaseModel

app = FastAPI()


class PontoEntrada(BaseModel):
    embedding: list[float]  # 128 floats
    id: str                 # nome da pessoa (até 100 caracteres)


@app.post("/construir-arvore")
def constroi_arvore():
    lib.kdtree_construir()
    return {"mensagem": "Árvore KD inicializada com sucesso."}

@app.post("/inserir")
def inserir(ponto: PontoEntrada):
    if len(ponto.embedding) != 128:
        return {"erro": "O vetor de embedding deve conter exatamente 128 floats."}

    emb_array = (c_float * 128)(*ponto.embedding)
    id_bytes = ponto.id.encode('utf-8')[:99]
    novo_ponto = TReg(embedding=emb_array, id=id_bytes)
    lib.inserir_ponto(novo_ponto)
    return {"mensagem": f"Ponto '{ponto.id}' inserido com sucesso."}


@app.post("/buscar")
def buscar(ponto: PontoEntrada):
    if len(ponto.embedding) != 128:
        return {"erro": "O vetor de embedding deve conter exatamente 128 floats."}

    emb_array = (c_float * 128)(*ponto.embedding)
    id_bytes = ponto.id.encode('utf-8')[:99]
    query = TReg(embedding=emb_array, id=id_bytes)

    arv = lib.get_tree()
    resultado = lib.buscar_mais_proximo(arv, query)

    return {
        "id": resultado.id.decode('utf-8').strip(),
        "embedding": list(resultado.embedding)
    }

