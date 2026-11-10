"""Cálculos compartilhados, sem leitura ou gravação de arquivos."""

import numpy as np


def encontrar_banda_mais_proxima(wavelengths, target_wavelength):
    """Encontra o índice da banda de comprimento de onda mais próximo do alvo."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("Informe uma lista não vazia de comprimentos de onda.")
    if not np.isfinite(wavelengths).all() or not np.isfinite(target_wavelength):
        raise ValueError("Os comprimentos de onda devem ser finitos.")
    return int(np.argmin(np.abs(wavelengths - target_wavelength)))


def mascara_pixels_validos(dados_pixels, nodata_val=-9999):
    """Seleciona espectros sem nodata e com soma estritamente positiva."""
    dados_pixels = np.asarray(dados_pixels)
    if dados_pixels.ndim != 2 or dados_pixels.shape[1] == 0:
        raise ValueError("Os pixels devem ter formato (pixels, bandas), com bandas.")
    return (dados_pixels != nodata_val).all(axis=1) & (dados_pixels.sum(axis=1) > 0)


def normalizar_por_percentil(
    dados, percentil_inferior=2, percentil_superior=98, mascara_referencia=None
):
    """Recorta e normaliza pelo percentil global ou pelos pixels de referência.

    No RGB, usa os percentis 2 e 98 de todos os canais juntos. Nos mapas de
    anomalia, ``percentil_inferior=None`` mantém o limite inferior fixo em zero.
    Uma máscara permite calcular os limites apenas sobre a terra no refinamento.
    """
    dados = np.asarray(dados)
    if mascara_referencia is None:
        referencia = dados
    else:
        mascara_referencia = np.asarray(mascara_referencia, dtype=bool)
        if mascara_referencia.shape != dados.shape:
            raise ValueError("A máscara de referência deve ter o formato dos dados.")
        referencia = dados[mascara_referencia]
    if referencia.size == 0:
        raise ValueError("Não há pixels de referência para calcular o percentil.")

    if percentil_inferior is None:
        minimo = 0
        maximo = np.percentile(referencia, percentil_superior)
    else:
        minimo, maximo = np.percentile(
            referencia, (percentil_inferior, percentil_superior)
        )
    if not np.isfinite([minimo, maximo]).all():
        raise ValueError("Os limites de contraste devem ser finitos.")
    if maximo < minimo:
        raise ValueError("O limite superior de contraste deve ser maior que o inferior.")
    # Uma imagem sem variação não tem contraste; evita a divisão por zero.
    if maximo == minimo:
        return np.zeros(dados.shape, dtype=float)
    return (np.clip(dados, minimo, maximo) - minimo) / (maximo - minimo)
