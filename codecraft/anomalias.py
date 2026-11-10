"""Treinamento do autoencoder e detecção pelo erro de reconstrução."""

import os
from pathlib import Path

import numpy as np

from ._arquivos import abrir_envi, preparar_saida
from .processamento import mascara_pixels_validos, normalizar_por_percentil


def treinar_autoencoder(x_train, epocas=50, semente=42):
    """Constrói e treina o modelo original com semente e épocas configuráveis."""
    if epocas <= 0:
        raise ValueError("O número de épocas deve ser positivo.")
    if not 0 <= semente < 2**32:
        raise ValueError("A semente deve estar entre 0 e 4294967295.")
    # TensorFlow é necessário somente para o treinamento.
    # Desativa logs informativos para uma saída mais limpa.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    keras = tf.keras
    keras.utils.set_random_seed(semente)
    tf.config.experimental.enable_op_determinism()
    num_bands = x_train.shape[1]
    encoding_dim = 16
    autoencoder = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(num_bands,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(encoding_dim, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(num_bands, activation="sigmoid"),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")
    print(f"Treinando por {epocas} épocas (semente {semente})...")
    autoencoder.fit(x_train, x_train, epochs=epocas, batch_size=256, shuffle=True, verbose=2)
    return autoencoder


def treinar_e_detectar_anomalias(
    caminho_hdr_treino, caminho_hdr_analise, caminho_saida_tif, caminho_saida_png,
    epocas=50, semente=42,
):
    """Treina na área preservada e salva o erro real em TIF e o contraste em PNG."""
    import matplotlib.pyplot as plt
    import rasterio
    from sklearn.preprocessing import MinMaxScaler

    # --- 1. PREPARAÇÃO DOS DADOS DE TREINAMENTO (ÁREA DE PRESERVAÇÃO) ---
    print("--- Fase de Treinamento ---")
    _, referencia_treino = abrir_envi(caminho_hdr_treino)
    _, referencia_analise = abrir_envi(caminho_hdr_analise)
    if referencia_treino.nbands != referencia_analise.nbands:
        raise ValueError("As imagens de treino e análise devem ter o mesmo número de bandas.")
    img_treino = referencia_treino.load()
    h_t, w_t, num_bands = img_treino.shape
    dados_pixels_treino = img_treino.reshape((h_t * w_t, num_bands))
    mask_validos_treino = mascara_pixels_validos(dados_pixels_treino)
    dados_treino_validos = dados_pixels_treino[mask_validos_treino]
    if len(dados_treino_validos) == 0:
        raise ValueError("A imagem de treino não contém pixels válidos.")
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(dados_treino_validos)
    print(f"Dados de treinamento prontos: {len(x_train)} pixels saudáveis.")

    # --- 2. CONSTRUÇÃO E TREINAMENTO DO MODELO AUTOENCODER ---
    autoencoder = treinar_autoencoder(x_train, epocas=epocas, semente=semente)

    # --- 3. PREPARAÇÃO DOS DADOS DE ANÁLISE (NIQUELÂNDIA) ---
    print("--- Fase de Detecção ---")
    img_analise = referencia_analise.load()
    h_a, w_a, _ = img_analise.shape
    dados_pixels_analise = img_analise.reshape((h_a * w_a, num_bands))
    mask_validos_analise = mascara_pixels_validos(dados_pixels_analise)
    dados_analise_validos = dados_pixels_analise[mask_validos_analise]
    if len(dados_analise_validos) == 0:
        raise ValueError("A imagem de análise não contém pixels válidos.")
    dados_analise_normalizados = scaler.transform(dados_analise_validos)

    # --- 4. DETECÇÃO DE ANOMALIAS NA ÁREA DE ANÁLISE ---
    pixels_reconstruidos = autoencoder.predict(dados_analise_normalizados)
    mse_erro = np.mean(np.power(dados_analise_normalizados - pixels_reconstruidos, 2), axis=1)
    print(
        f"Erro de reconstrução: Min={np.min(mse_erro):.6f}, "
        f"Max={np.max(mse_erro):.6f}, Média={np.mean(mse_erro):.6f}"
    )
    mapa_anomalia_final = np.full(h_a * w_a, 0.0)
    mapa_anomalia_final[mask_validos_analise] = mse_erro
    mapa_anomalia_final = mapa_anomalia_final.reshape((h_a, w_a))

    # Normalizar de zero ao 98º percentil acentua o contraste das anomalias.
    mapa_anomalia_norm = normalizar_por_percentil(mapa_anomalia_final, percentil_inferior=None)

    # --- 5. SALVANDO OS RESULTADOS ---
    caminho_raw_analise = Path(caminho_hdr_analise).with_suffix(".raw")
    with rasterio.open(caminho_raw_analise) as src_ref:
        transform, crs = src_ref.transform, src_ref.crs
    with rasterio.open(
        preparar_saida(caminho_saida_tif), "w", driver="GTiff",
        height=h_a, width=w_a, count=1, dtype=rasterio.float32,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(mapa_anomalia_final.astype(rasterio.float32), 1)  # Erro real no TIF.
    plt.imsave(preparar_saida(caminho_saida_png), mapa_anomalia_norm, cmap="jet")
    print(f"Mapa de erros salvo em: '{caminho_saida_tif}'.")
    print(f"Visualização salva em: '{caminho_saida_png}'.")
