import numpy as np
import rasterio
from spectral import envi
import matplotlib.pyplot as plt

def encontrar_banda_mais_proxima(wavelengths, target_wavelength):
    """Encontra o índice da banda cujo comprimento de onda é mais próximo do alvo."""
    idx = np.argmin(np.abs(np.array(wavelengths) - target_wavelength))
    return idx

def refinar_mapa_anomalia(caminho_hdr_original, caminho_tif_anomalia, caminho_saida_final_png):
    """
    Mascara corpos d'água em um mapa de anomalias e reescala o contraste para
    revelar anomalias sutis na vegetação.
    """
    try:
        print("--- Iniciando Refinamento do Mapa de Anomalias ---")
        
        # --- 1. CALCULAR A MÁSCARA DE ÁGUA USANDO NDWI ---
        print("Passo 1: Calculando Índice de Água (NDWI) para criar máscara...")
        
        hdr = envi.read_envi_header(caminho_hdr_original)
        img = envi.open(caminho_hdr_original)

        # NDWI usa as bandas Verde e Infravermelho Próximo (NIR)
        try:
            wavelengths = [float(w) for w in hdr['wavelength']]
            green_idx = encontrar_banda_mais_proxima(wavelengths, 550) # Verde
            nir_idx = encontrar_banda_mais_proxima(wavelengths, 860)   # NIR
            print(f"Usando bandas: Verde (idx {green_idx}) e NIR (idx {nir_idx}) para o NDWI.")
        except KeyError:
            print("AVISO: 'wavelength' não encontrado. Usando índices de banda padrão para EMIT.")
            green_idx, nir_idx = 35, 85 # Índices aproximados para Verde e NIR no EMIT

        green_band = img.read_band(green_idx).astype(float)
        nir_band = img.read_band(nir_idx).astype(float)
        
        # Evitar divisão por zero
        np.seterr(divide='ignore', invalid='ignore')
        ndwi = (green_band - nir_band) / (green_band + nir_band)
        
        # Criar a máscara: pixels com NDWI > 0.2 são considerados água
        mascara_agua = ndwi > 0.2
        print(f"Máscara de água criada. {np.count_nonzero(mascara_agua)} pixels de água encontrados.")

        # --- 2. APLICAR A MÁSCARA E REESCALAR O CONTRASTE ---
        print("Passo 2: Aplicando máscara e reescalando contraste...")
        
        with rasterio.open(caminho_tif_anomalia) as src:
            mapa_anomalia = src.read(1)
            
        # Aplicar a máscara: onde for água, o valor da anomalia se torna 0
        mapa_anomalia_mascarado = np.where(mascara_agua, 0, mapa_anomalia)
        
        # Criar uma máscara dos pixels de terra para o cálculo do percentil
        mascara_terra = ~mascara_agua
        
        # Recalcular o limite de contraste APENAS nos pixels de terra
        vmax = np.percentile(mapa_anomalia_mascarado[mascara_terra], 98)
        print(f"Novo limite de contraste (98º percentil na terra): {vmax:.6f}")

        # Normalizar o mapa com o novo limite
        mapa_anomalia_refinado = np.clip(mapa_anomalia_mascarado, 0, vmax) / vmax
        
        # --- 3. SALVAR O RESULTADO FINAL ---
        print("Passo 3: Salvando resultado final...")
        plt.imsave(caminho_saida_final_png, mapa_anomalia_refinado, cmap='jet')
        print(f"\nSucesso! Mapa final refinado salvo em: '{caminho_saida_final_png}'")
        
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado. Verifique os caminhos.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# --- COMO USAR O SCRIPT ---
if __name__ == '__main__':
    # 1. O arquivo .hdr original da área de análise (Niquelândia)
    arquivo_hdr_niquelandia = 'C:/Users/hugo/OneDrive/Documentos/codecraft/EMIT_L2A_reflectance_convertido.hdr'
    
    # 2. O mapa de anomalias .tif gerado pelo script de Deep Learning
    mapa_anomalia_dl_tif = 'C:/Users/hugo/OneDrive/Documentos/codecraft/mapa_anomalia_niquelandia_dl.tif'
    
    # 3. O nome da imagem PNG de saída final e refinada
    arquivo_png_final = 'mapa_anomalia_FINAL_refinado.png'
    
    refinar_mapa_anomalia(arquivo_hdr_niquelandia, mapa_anomalia_dl_tif, arquivo_png_final)