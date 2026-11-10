"""Conversão do NetCDF EMIT L2A Reflectance para ENVI BSQ."""

import numpy as np

from ._arquivos import preparar_saida, verificar_entrada


def montar_cabecalho_envi(amostras, linhas, bandas, wavelengths=None):
    """Monta o cabeçalho ENVI para dados float32, little-endian, em ordem BSQ."""
    if min(amostras, linhas, bandas) <= 0:
        raise ValueError("As dimensões do cabeçalho ENVI devem ser positivas.")
    header_lines = [
        "ENVI",
        "description = {Arquivo EMIT L2A Reflectance (dimensões corrigidas)}",
        f"samples = {amostras}",
        f"lines   = {linhas}",
        f"bands   = {bandas}",
        "header offset = 0",
        "file type = ENVI Standard",
        "data type = 4",  # float32
        "interleave = bsq",
        "byte order = 0",
        "data ignore value = -9999",
    ]
    # Adicionar comprimentos de onda, se disponíveis.
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths)
        if wavelengths.shape != (bandas,):
            raise ValueError("Deve haver um comprimento de onda para cada banda.")
        wavelengths_str = ", ".join(map(str, np.round(wavelengths, 2)))
        header_lines.append(f"wavelength = {{{wavelengths_str}}}")
    return "\n".join(header_lines) + "\n"


def converter_emit_para_envi(caminho_arquivo_nc, caminho_saida_base):
    """Converte reflectance para .hdr + .raw, identificando as dimensões por nome."""
    import xarray as xr

    caminho_arquivo_nc = verificar_entrada(caminho_arquivo_nc)
    print(f"Iniciando a conversão de: '{caminho_arquivo_nc}'...")
    # O gerenciador de contexto fecha o NetCDF também em caso de erro.
    with xr.open_dataset(caminho_arquivo_nc) as dataset:
        if "reflectance" not in dataset.variables:
            raise ValueError("A variável 'reflectance' não foi encontrada no NetCDF.")
        imagem_data = dataset["reflectance"]
        if imagem_data.ndim != 3:
            raise ValueError("A variável 'reflectance' deve ter três dimensões.")

        # Identificar os eixos pelos nomes evita a troca de linhas e bandas.
        dimensoes = ("bands", "downtrack", "crosstrack")
        if not all(dim in imagem_data.dims for dim in dimensoes):
            # Plano B: a dimensão com 285 elementos deve ser a de bandas EMIT.
            indices = np.flatnonzero(np.array(imagem_data.shape) == 285)
            if len(indices) != 1 or indices[0] not in (0, 2):
                raise ValueError(
                    "Não foi possível determinar as dimensões: esperadas bands, "
                    "downtrack e crosstrack, ou 285 bandas no primeiro ou último eixo."
                )
            print("Aviso: inferindo a ordem das dimensões pelas 285 bandas EMIT.")
            if indices[0] == 0:
                dimensoes = imagem_data.dims
            else:
                dimensoes = (imagem_data.dims[2], *imagem_data.dims[:2])

        # Para BSQ (Band Sequential), a ordem é (bands, lines, samples).
        imagem_bsq = imagem_data.transpose(*dimensoes)
        bandas, linhas, amostras = imagem_bsq.shape
        print(f"Dimensões: {amostras} (amostras) x {linhas} (linhas) x {bandas} (bandas)")
        wavelengths = (
            imagem_data.coords["wavelengths"].values
            if "wavelengths" in imagem_data.coords else None
        )
        header = montar_cabecalho_envi(amostras, linhas, bandas, wavelengths)
        # O tipo e a ordem dos bytes precisam corresponder ao cabeçalho.
        dados_numpy = np.asarray(imagem_bsq.values, dtype="<f4")
        caminho_hdr = preparar_saida(f"{caminho_saida_base}.hdr")
        caminho_raw = preparar_saida(f"{caminho_saida_base}.raw")
        caminho_hdr.write_text(header, encoding="utf-8")
        with caminho_raw.open("wb") as arquivo:
            dados_numpy.tofile(arquivo)
    print(f"Conversão concluída: '{caminho_hdr}' e '{caminho_raw}'.")
