"""Composição RGB visível com aprimoramento de contraste."""

import numpy as np

from ._arquivos import abrir_envi, preparar_saida
from .processamento import encontrar_banda_mais_proxima, normalizar_por_percentil


def compor_rgb(rgb_data, nodata_val=-9999):
    """Substitui nodata por zero e aplica os percentis globais 2 e 98."""
    rgb_data = np.array(rgb_data, copy=True)
    if rgb_data.ndim != 3 or rgb_data.shape[-1] != 3:
        raise ValueError("Os dados RGB devem ter formato (linhas, amostras, 3).")
    rgb_data[rgb_data == nodata_val] = 0
    rgb_normalized = normalizar_por_percentil(rgb_data)
    return (rgb_normalized * 255).astype(np.uint8)


def converter_raw_para_rgb(caminho_arquivo_hdr, caminho_saida_rgb):
    """Lê as bandas RGB de uma imagem ENVI e salva a composição visível."""
    import matplotlib.pyplot as plt

    hdr, img = abrir_envi(caminho_arquivo_hdr)
    if "wavelength" in hdr:
        wavelengths = [float(w) for w in hdr["wavelength"]]
        # Alvos de comprimento de onda para Vermelho, Verde e Azul, em nm.
        indices = [encontrar_banda_mais_proxima(wavelengths, alvo) for alvo in (650, 550, 450)]
        print(f"Bandas RGB selecionadas por comprimento de onda: {indices}.")
    else:
        # Estimativas originais para o sensor EMIT (285 bandas).
        print("Aviso: 'wavelength' ausente. Usando índices RGB padrão do EMIT.")
        indices = [40, 25, 15]
    if max(indices) >= img.nbands:
        raise ValueError("A imagem não contém as bandas necessárias para compor o RGB.")
    rgb_data = img.read_bands(indices)
    rgb_final = compor_rgb(rgb_data, float(hdr.get("data ignore value", -9999)))
    plt.imsave(preparar_saida(caminho_saida_rgb), rgb_final)
    print(f"Imagem RGB visível salva em: '{caminho_saida_rgb}'.")
