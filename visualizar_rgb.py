import numpy as np
from spectral import envi
import matplotlib.pyplot as plt

def encontrar_banda_mais_proxima(wavelengths, target_wavelength):
    """Encontra o índice da banda cujo comprimento de onda é mais próximo do alvo."""
    idx = np.argmin(np.abs(np.array(wavelengths) - target_wavelength))
    return idx

def converter_raw_para_rgb(caminho_arquivo_hdr, caminho_saida_rgb):
    """
    Lê um arquivo hiperespectral ENVI (.raw + .hdr) e o converte para uma imagem
    RGB visível (.png) com aprimoramento de contraste.
    Este script agora lida com a ausência de dados de comprimento de onda.
    """
    try:
        # 1. Ler o cabeçalho para obter metadados
        hdr = envi.read_envi_header(caminho_arquivo_hdr)
        img = envi.open(caminho_arquivo_hdr)

        # 2. TENTAR encontrar as bandas RGB usando comprimentos de onda
        try:
            wavelengths = [float(w) for w in hdr['wavelength']]
            
            red_target = 650
            green_target = 550
            blue_target = 450
            
            red_idx = encontrar_banda_mais_proxima(wavelengths, red_target)
            green_idx = encontrar_banda_mais_proxima(wavelengths, green_target)
            blue_idx = encontrar_banda_mais_proxima(wavelengths, blue_target)

            print("--- Seleção de Bandas por Comprimento de Onda ---")
            print(f"Banda Vermelha (R): Índice {red_idx} @ {wavelengths[red_idx]:.2f} nm")
            print(f"Banda Verde   (G): Índice {green_idx} @ {wavelengths[green_idx]:.2f} nm")
            print(f"Banda Azul    (B): Índice {blue_idx} @ {wavelengths[blue_idx]:.2f} nm")

        except KeyError:
            # SE FALHAR (KeyError: 'wavelength'), usar bandas padrão para EMIT
            print("\nAVISO: Informação de 'wavelength' não encontrada no arquivo .hdr.")
            print("Usando índices de banda padrão para o sensor EMIT (285 bandas).")
            
            # Estas são estimativas seguras para o sensor EMIT
            red_idx = 40
            green_idx = 25
            blue_idx = 15
            
            print("--- Seleção de Bandas por Índices Padrão ---")
            print(f"Banda Vermelha (R): Índice {red_idx}")
            print(f"Banda Verde   (G): Índice {green_idx}")
            print(f"Banda Azul    (B): Índice {blue_idx}")

        # 3. Ler os dados das bandas RGB selecionadas
        rgb_data = img.read_bands([red_idx, green_idx, blue_idx])
        
        nodata_val = float(hdr.get('data ignore value', -9999))
        rgb_data[rgb_data == nodata_val] = 0

        # 4. Aprimoramento de Contraste
        p2, p98 = np.percentile(rgb_data, (2, 98))
        rgb_stretched = np.clip(rgb_data, p2, p98)
        rgb_normalized = (rgb_stretched - p2) / (p98 - p2)
        rgb_final = (rgb_normalized * 255).astype(np.uint8)

        # 5. Salvar a imagem RGB final
        plt.imsave(caminho_saida_rgb, rgb_final)
        print(f"\nSucesso! Imagem RGB visível salva em: '{caminho_saida_rgb}'")

    except FileNotFoundError:
        print(f"Erro: Arquivo de cabeçalho não encontrado em '{caminho_arquivo_hdr}'")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# --- COMO USAR O SCRIPT ---
if __name__ == '__main__':
    # Verifique se este é o nome correto do seu arquivo .hdr
    arquivo_hdr_entrada = 'C:/Users/hugo/OneDrive/Documentos/codecraft/EMIT_L2A_reflectance_convertido.hdr'
    
    # Defina o nome para a imagem de saída
    arquivo_png_saida = 'imagem_rgb_visivel.png'
    
    converter_raw_para_rgb(arquivo_hdr_entrada, arquivo_png_saida)