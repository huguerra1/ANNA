"""Máscara de água por NDWI e contraste das anomalias em terra."""

import numpy as np

from ._arquivos import abrir_envi, preparar_saida, verificar_entrada
from .processamento import encontrar_banda_mais_proxima, normalizar_por_percentil


def calcular_ndwi(green_band, nir_band):
    """Calcula (Verde - NIR) / (Verde + NIR), preservando NaN e infinito."""
    green_band = np.asarray(green_band, dtype=float)
    nir_band = np.asarray(nir_band, dtype=float)
    if green_band.shape != nir_band.shape:
        raise ValueError("As bandas Verde e NIR devem ter o mesmo formato.")
    # Evitar avisos de divisão por zero sem alterar a configuração global NumPy.
    with np.errstate(divide="ignore", invalid="ignore"):
        return (green_band - nir_band) / (green_band + nir_band)


def criar_mascara_agua(ndwi, limiar=0.2):
    """Considera água apenas os pixels com NDWI estritamente maior que 0,2."""
    return np.asarray(ndwi) > limiar


def refinar_contraste(mapa_anomalia, mascara_agua):
    """Zera a água e normaliza de zero ao percentil 98 calculado só na terra."""
    mapa_anomalia = np.asarray(mapa_anomalia)
    mascara_agua = np.asarray(mascara_agua, dtype=bool)
    if mapa_anomalia.shape != mascara_agua.shape:
        raise ValueError("O mapa de anomalias e a máscara de água devem ter o mesmo formato.")
    mapa_anomalia_mascarado = np.where(mascara_agua, 0, mapa_anomalia)
    mascara_terra = ~mascara_agua
    return normalizar_por_percentil(
        mapa_anomalia_mascarado, percentil_inferior=None,
        mascara_referencia=mascara_terra,
    )


def refinar_mapa_anomalia(caminho_hdr_original, caminho_tif_anomalia, caminho_saida_final_png):
    """Lê a imagem original e o erro em TIF e salva o mapa refinado em PNG."""
    import matplotlib.pyplot as plt
    import rasterio

    print("--- Iniciando Refinamento do Mapa de Anomalias ---")
    verificar_entrada(caminho_tif_anomalia)
    hdr, img = abrir_envi(caminho_hdr_original)
    # --- 1. CALCULAR A MÁSCARA DE ÁGUA USANDO NDWI ---
    # NDWI usa as bandas Verde e Infravermelho Próximo (NIR).
    if "wavelength" in hdr:
        wavelengths = [float(w) for w in hdr["wavelength"]]
        green_idx = encontrar_banda_mais_proxima(wavelengths, 550)
        nir_idx = encontrar_banda_mais_proxima(wavelengths, 860)
    else:
        print("Aviso: 'wavelength' ausente. Usando índices NDWI padrão do EMIT.")
        green_idx, nir_idx = 35, 85  # Índices aproximados originais para EMIT.
    if max(green_idx, nir_idx) >= img.nbands:
        raise ValueError("A imagem não contém as bandas necessárias para calcular o NDWI.")
    ndwi = calcular_ndwi(img.read_band(green_idx), img.read_band(nir_idx))
    mascara_agua = criar_mascara_agua(ndwi)
    print(f"Máscara de água criada: {np.count_nonzero(mascara_agua)} pixels de água.")

    # --- 2. APLICAR A MÁSCARA E REESCALAR O CONTRASTE ---
    with rasterio.open(caminho_tif_anomalia) as src:
        mapa_anomalia = src.read(1)
    mapa_anomalia_refinado = refinar_contraste(mapa_anomalia, mascara_agua)

    # --- 3. SALVAR O RESULTADO FINAL ---
    plt.imsave(preparar_saida(caminho_saida_final_png), mapa_anomalia_refinado, cmap="jet")
    print(f"Mapa final refinado salvo em: '{caminho_saida_final_png}'.")
