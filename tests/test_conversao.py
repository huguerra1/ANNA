"""Cabeçalho e conversão de NetCDFs sintéticos para ENVI BSQ."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest

import numpy as np
import xarray as xr

from codecraft.conversao import converter_emit_para_envi, montar_cabecalho_envi
from tests.temporarios import diretorio_temporario


class TestCabecalhoEnvi(unittest.TestCase):
    def test_conteudo_e_comprimentos_de_onda(self):
        header = montar_cabecalho_envi(4, 2, 3, [450.123, 550.456, 860.789])
        self.assertEqual(header, "\n".join([
            "ENVI",
            "description = {Arquivo EMIT L2A Reflectance (dimensões corrigidas)}",
            "samples = 4",
            "lines   = 2",
            "bands   = 3",
            "header offset = 0",
            "file type = ENVI Standard",
            "data type = 4",
            "interleave = bsq",
            "byte order = 0",
            "data ignore value = -9999",
            "wavelength = {450.12, 550.46, 860.79}",
            "",
        ]))

    def test_comprimentos_de_onda_opcionais(self):
        self.assertNotIn("wavelength", montar_cabecalho_envi(4, 2, 3))

    def test_quantidade_de_comprimentos_de_onda_incompativel(self):
        with self.assertRaisesRegex(ValueError, "cada banda"):
            montar_cabecalho_envi(4, 2, 3, [450, 550])


class TestConversaoNetCDF(unittest.TestCase):
    def test_dimensoes_nomeadas_em_diferentes_ordens_e_raw_bsq(self):
        # Cada posição tem valor distinto, tornando trocas de eixo detectáveis.
        dados_bsq = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        dados_bsq[0, 0, 0] = -9999
        dimensoes = ("bands", "downtrack", "crosstrack")
        for ordem in [dimensoes, ("downtrack", "crosstrack", "bands"), ("crosstrack", "bands", "downtrack")]:
            with self.subTest(ordem=ordem), diretorio_temporario() as temporario:
                pasta = Path(temporario)
                entrada = pasta / "entrada.nc"
                saida = pasta / "subpasta" / "emas"
                array = xr.DataArray(
                    dados_bsq, dims=dimensoes,
                    coords={"wavelengths": ("bands", [450.123, 550.456, 860.789])},
                ).transpose(*ordem)
                with xr.Dataset({"reflectance": array}) as dataset:
                    dataset.to_netcdf(entrada, engine="netcdf4")
                with redirect_stdout(io.StringIO()):
                    converter_emit_para_envi(entrada, saida)

                header = saida.with_suffix(".hdr").read_text(encoding="utf-8")
                self.assertEqual(header, montar_cabecalho_envi(4, 2, 3, [450.123, 550.456, 860.789]))
                caminho_raw = saida.with_suffix(".raw")
                self.assertEqual(caminho_raw.stat().st_size, 24 * 4)
                # A sequência exata é banda a banda (BSQ), não BIL ou BIP.
                self.assertEqual(caminho_raw.read_bytes(), dados_bsq.astype("<f4").tobytes(order="C"))
                np.testing.assert_array_equal(
                    np.fromfile(caminho_raw, dtype="<f4").reshape(3, 2, 4), dados_bsq
                )

    def test_float64_e_gravado_como_float32_do_cabecalho(self):
        with diretorio_temporario() as temporario:
            pasta = Path(temporario)
            entrada = pasta / "entrada.nc"
            dados = np.array([[[0.25, 0.5]]], dtype=np.float64)
            with xr.Dataset({"reflectance": (("bands", "downtrack", "crosstrack"), dados)}) as dataset:
                dataset.to_netcdf(entrada, engine="netcdf4")
            with redirect_stdout(io.StringIO()):
                converter_emit_para_envi(entrada, pasta / "saida")
            self.assertEqual((pasta / "saida.raw").read_bytes(), dados.astype("<f4").tobytes())

    def test_variavel_ausente_gera_erro_e_fecha_netcdf(self):
        with diretorio_temporario() as temporario:
            pasta = Path(temporario)
            entrada = pasta / "entrada.nc"
            with xr.Dataset({"outra": ("x", [1.])}) as dataset:
                dataset.to_netcdf(entrada, engine="netcdf4")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "'reflectance'"):
                converter_emit_para_envi(entrada, pasta / "saida")
            self.assertFalse((pasta / "saida.hdr").exists())
            entrada.unlink()  # No Windows também verifica que o arquivo foi fechado.

    def test_dimensoes_desconhecidas_tem_erro_claro(self):
        with diretorio_temporario() as temporario:
            pasta = Path(temporario)
            entrada = pasta / "entrada.nc"
            with xr.Dataset({"reflectance": (("z", "y", "x"), np.zeros((3, 2, 4)))}) as dataset:
                dataset.to_netcdf(entrada, engine="netcdf4")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "dimensões"):
                converter_emit_para_envi(entrada, pasta / "saida")


if __name__ == "__main__":
    unittest.main()
