"""Diretórios temporários com permissões herdadas, inclusive no Windows."""

from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4


@contextmanager
def diretorio_temporario():
    # Python 3.13 cria TemporaryDirectory com modo 0o700; em alguns ambientes
    # Windows restritos, essa ACL impede o próprio processo de acessar a pasta.
    raiz = Path(__file__).resolve().parent
    caminho = raiz / f"temporario_{uuid4().hex}"
    caminho.mkdir()
    try:
        yield caminho
    finally:
        # A limpeza recursiva deve permanecer dentro da pasta de testes.
        if caminho.resolve().parent != raiz:
            raise ValueError("O diretório temporário deve estar dentro da pasta de testes.")
        shutil.rmtree(caminho)
