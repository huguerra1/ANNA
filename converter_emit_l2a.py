import xarray as xr
import numpy as np
import sys

def converter_emit_para_envi(caminho_arquivo_nc, caminho_saida_base):
    """
    Versão CORRIGIDA: Converte um arquivo NetCDF EMIT L2A para o formato ENVI,
    identificando as dimensões corretamente pelos seus nomes.
    """
    print(f"Iniciando a conversão (versão corrigida) de: '{caminho_arquivo_nc}'...")

    dataset = None
    try:
        # 1. Abrir o arquivo NetCDF com xarray
        dataset = xr.open_dataset(caminho_arquivo_nc)

        if 'reflectance' not in dataset.variables:
            print("Erro: A variável 'reflectance' não foi encontrada.")
            return
        
        imagem_data = dataset['reflectance']

        # 2. Extrair metadados identificando as dimensões pelos nomes
        # Esta é a correção principal para evitar a troca de eixos.
        try:
            # A ordem esperada é (bands, downtrack, crosstrack)
            bandas = imagem_data.sizes['bands']
            linhas = imagem_data.sizes['downtrack']
            amostras = imagem_data.sizes['crosstrack']
        except KeyError:
            # Plano B se os nomes forem diferentes (ex: x, y, band)
            print("Aviso: Nomes de dimensão padrão não encontrados, tentando inferir...")
            # Encontra a dimensão com 285, que deve ser a de bandas
            band_dim_index = np.where(np.array(imagem_data.shape) == 285)[0][0]
            
            if band_dim_index == 0: # (bands, lines, samples)
                bandas, linhas, amostras = imagem_data.shape
            elif band_dim_index == 2: # (lines, samples, bands)
                linhas, amostras, bandas = imagem_data.shape
            else:
                raise ValueError("Não foi possível determinar a ordem das dimensões da imagem.")
        
        print(f"Dimensões corretas: {amostras} (amostras) x {linhas} (linhas) x {bandas} (bandas)")

        # 3. Garantir que os dados estejam na ordem correta para salvar em BSQ
        # A ordem para BSQ (Band Sequential) deve ser (bands, lines, samples)
        dados_numpy = imagem_data.transpose('bands', 'downtrack', 'crosstrack').values

        # 4. Criar o cabeçalho (.hdr) com os valores corretos
        tipo_dado_envi = 4 # float32
        byte_order = 0
        
        caminho_saida_hdr = f"{caminho_saida_base}.hdr"
        
        header_lines = [
            "ENVI",
            f"description = {{Arquivo EMIT L2A Reflectance (dimensões corrigidas)}}",
            f"samples = {amostras}",
            f"lines   = {linhas}",
            f"bands   = {bandas}",
            "header offset = 0",
            "file type = ENVI Standard",
            f"data type = {tipo_dado_envi}",
            "interleave = bsq",
            f"byte order = {byte_order}",
            "data ignore value = -9999"
        ]
        
        # Adicionar comprimentos de onda, se disponíveis
        if 'wavelengths' in imagem_data.coords:
            wavelengths = imagem_data.coords['wavelengths'].values
            wavelengths_str = ", ".join(map(str, np.round(wavelengths, 2)))
            header_lines.append(f"wavelength = {{{wavelengths_str}}}")

        header = "\n".join(header_lines) + "\n"

        with open(caminho_saida_hdr, 'w') as f:
            f.write(header)
        print(f"Arquivo de cabeçalho (.hdr) corrigido salvo em: '{caminho_saida_hdr}'")

        # 5. Salvar o arquivo de dados brutos (.raw) com a ordem correta
        caminho_saida_raw = f"{caminho_saida_base}.raw"
        with open(caminho_saida_raw, 'wb') as f:
            dados_numpy.tofile(f)
        print(f"Arquivo de dados brutos (.raw) corrigido salvo em: '{caminho_saida_raw}'")
        print("\nConversão concluída com sucesso!")

    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada '{caminho_arquivo_nc}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    finally:
        if dataset:
            dataset.close()

# --- COMO USAR O SCRIPT ---
if __name__ == '__main__':
    arquivo_nc_entrada = 'C:/Users/hugo/OneDrive/Documentos/codecraft/EMIT_L2A_RFL_001_20250902T123826_2524508_037.nc'
    arquivo_raw_saida = 'C:/Users/hugo/OneDrive/Documentos/codecraft/parque_das_emas_convertido'
    converter_emit_para_envi(arquivo_nc_entrada, arquivo_raw_saida)