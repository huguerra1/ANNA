"""Operações de arquivo compartilhadas pelos comandos."""

from pathlib import Path


def verificar_entrada(caminho):
    """Verifica a existência da entrada e retorna seu caminho."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: '{caminho}'.")
    return caminho


def preparar_saida(caminho):
    """Cria os diretórios necessários para uma saída."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho


def abrir_envi(caminho_hdr):
    """Abre o cabeçalho e a imagem ENVI, com contexto nos erros do Spectral."""
    from spectral import envi
    from spectral.utilities.errors import SpyException

    caminho_hdr = verificar_entrada(caminho_hdr)
    try:
        hdr = envi.read_envi_header(str(caminho_hdr))
        imagem = envi.open(str(caminho_hdr))
    except SpyException as erro:
        raise ValueError(f"Não foi possível ler o ENVI '{caminho_hdr}': {erro}") from erro
    return hdr, imagem
