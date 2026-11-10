import numpy as np
import tensorflow as tf
from tensorflow import keras
from spectral import envi
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import rasterio
import os

# Desativa logs informativos do TensorFlow para uma saída mais limpa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def treinar_e_detectar_anomalias(caminho_hdr_treino, caminho_hdr_analise, caminho_saida_tif, caminho_saida_png):
    """
    Versão DEFINITIVA: Treina um Autoencoder com uma imagem de referência 'saudável'
    e usa um método de visualização com contraste acentuado para garantir que as
    anomalias sejam visíveis.
    """
    try:
        # --- 1. PREPARAÇÃO DOS DADOS DE TREINAMENTO (ÁREA DE PRESERVAÇÃO) ---
        print("--- Fase de Treinamento ---")
        print(f"Passo 1: Carregando dados de treinamento de: '{caminho_hdr_treino}'")
        img_treino = envi.open(caminho_hdr_treino).load()
        h_t, w_t, num_bands = img_treino.shape
        dados_pixels_treino = img_treino.reshape((h_t * w_t, num_bands))
        
        nodata_val = -9999
        mask_validos_treino = (dados_pixels_treino != nodata_val).all(axis=1) & (dados_pixels_treino.sum(axis=1) > 0)
        dados_treino_validos = dados_pixels_treino[mask_validos_treino]
        
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(dados_treino_validos)
        print(f"Dados de treinamento prontos: {len(x_train)} pixels saudáveis.")

        # --- 2. CONSTRUÇÃO E TREINAMENTO DO MODELO AUTOENCODER ---
        print("Passo 2: Construindo e treinando o modelo Autoencoder...")
        encoding_dim = 16
        autoencoder = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(num_bands,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(encoding_dim, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_bands, activation='sigmoid')
        ])
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # MUDANÇA 1: Aumentar as épocas de treinamento para 50
        print("Treinando por 50 épocas para maior especialização do modelo...")
        autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, verbose=2)
        print("Modelo treinado com sucesso!")

        # --- 3. PREPARAÇÃO DOS DADOS DE ANÁLISE (NIQUELÂNDIA) ---
        print("\n--- Fase de Detecção ---")
        print(f"Passo 3: Carregando dados de análise de: '{caminho_hdr_analise}'")
        img_analise = envi.open(caminho_hdr_analise).load()
        h_a, w_a, _ = img_analise.shape
        dados_pixels_analise = img_analise.reshape((h_a * w_a, num_bands))
        
        mask_validos_analise = (dados_pixels_analise != nodata_val).all(axis=1) & (dados_pixels_analise.sum(axis=1) > 0)
        dados_analise_validos = dados_pixels_analise[mask_validos_analise]
        dados_analise_normalizados = scaler.transform(dados_analise_validos)
        print("Dados de análise prontos.")

        # --- 4. DETECÇÃO DE ANOMALIAS NA ÁREA DE ANÁLISE ---
        print("Passo 4: Detectando anomalias na área de análise...")
        pixels_reconstruidos = autoencoder.predict(dados_analise_normalizados)
        mse_erro = np.mean(np.power(dados_analise_normalizados - pixels_reconstruidos, 2), axis=1)
        
        print(f"Diagnóstico do erro de reconstrução: Min={np.min(mse_erro):.6f}, Max={np.max(mse_erro):.6f}, Média={np.mean(mse_erro):.6f}")

        mapa_anomalia_final = np.full(h_a * w_a, 0.0)
        mapa_anomalia_final[mask_validos_analise] = mse_erro
        mapa_anomalia_final = mapa_anomalia_final.reshape((h_a, w_a))
        
        # --- MUDANÇA 2: VISUALIZAÇÃO COM CONTRASTE ACENTUADO ---
        # Em vez de normalizar pelo máximo, normalizamos pelo 98º percentil.
        # Isso força as anomalias a se destacarem em vermelho.
        vmax = np.percentile(mapa_anomalia_final, 98)
        print(f"Acentuando contraste: o valor máximo para visualização será {vmax:.6f} (98º percentil).")
        
        # Normaliza o mapa de 0 até o valor do percentil
        mapa_anomalia_norm = np.clip(mapa_anomalia_final, 0, vmax) / vmax
        
        # --- 5. SALVANDO OS RESULTADOS ---
        print("\nPasso 5: Salvando os mapas de anomalias...")
        
        caminho_raw_analise = caminho_hdr_analise.replace('.hdr', '.raw')
        with rasterio.open(caminho_raw_analise) as src_ref:
            transform = src_ref.transform
            crs = src_ref.crs

        with rasterio.open(
            caminho_saida_tif, 'w', driver='GTiff',
            height=h_a, width=w_a, count=1, dtype=rasterio.float32,
            crs=crs, transform=transform
        ) as dst:
            dst.write(mapa_anomalia_final.astype(rasterio.float32), 1) # Salva o erro real no TIF
        print(f"Mapa de anomalias GeoTIFF (com valores de erro reais) salvo em: '{caminho_saida_tif}'")

        plt.imsave(caminho_saida_png, mapa_anomalia_norm, cmap='jet')
        print(f"Visualização PNG (com contraste acentuado) salva em: '{caminho_saida_png}'")
        print("\nProcesso concluído com sucesso!")

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique os caminhos: '{caminho_hdr_treino}' e '{caminho_hdr_analise}'.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# --- COMO USAR O SCRIPT ---
if __name__ == '__main__':
    # 1. Defina o caminho para o arquivo .hdr da sua ÁREA DE PRESERVAÇÃO (treino)
    # Exemplo: Parque Nacional das Emas
    arquivo_hdr_treino = 'C:/Users/hugo/OneDrive/Documentos/codecraft/parque_das_emas_convertido.hdr'
    
    # 2. Defina o caminho para o arquivo .hdr da sua ÁREA DE ANÁLISE (alvo)
    # Exemplo: Niquelândia
    arquivo_hdr_analise = 'C:/Users/hugo/OneDrive/Documentos/codecraft/EMIT_L2A_reflectance_convertido.hdr'
    
    # 3. Defina os nomes dos arquivos de saída
    arquivo_tif_saida = 'mapa_anomalia_niquelandia_dl.tif'
    arquivo_png_saida = 'mapa_anomalia_niquelandia_dl_visual.png'
    
    treinar_e_detectar_anomalias(arquivo_hdr_treino, arquivo_hdr_analise, arquivo_tif_saida, arquivo_png_saida)