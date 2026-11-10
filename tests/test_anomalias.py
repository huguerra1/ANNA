"""Treino e detecção completos com cubos ENVI sintéticos pequenos."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from anna.anomalias import treinar_e_detectar_anomalias
from tests.temporarios import diretorio_temporario


class TestAnomalias(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow") is not None,
        "O teste de treino requer TensorFlow instalado.",
    )
    def test_treino_e_deteccao_com_anomalia_espectral(self):
        from matplotlib.image import imread
        import rasterio
        from rasterio.transform import from_origin
        from spectral import envi

        gerador = np.random.default_rng(42)
        espectro = np.linspace(0.2, 0.4, 20, dtype=np.float32)
        saudavel = (espectro + gerador.normal(0, 0.005, (12, 12, 20))).astype(np.float32)
        alvo = saudavel.copy()
        anomalia = np.zeros((12, 12), dtype=bool)
        anomalia[4:8, 4:8] = True
        alvo[anomalia] = np.linspace(0.9, 0.7, 20, dtype=np.float32)

        # O .raw BSQ e o cabeçalho com UTM são lidos tanto pelo Spectral
        # quanto pelo Rasterio, usado no fluxo para copiar a georreferência.
        metadados = {
            "map info": ["UTM", 1, 1, 500000, 8000000, 30, 30, 22, "South", "WGS-84"],
        }
        with diretorio_temporario() as temporario:
            pasta = Path(temporario)
            treino_hdr = pasta / "saudavel.hdr"
            alvo_hdr = pasta / "alvo.hdr"
            for caminho, cubo in ((treino_hdr, saudavel), (alvo_hdr, alvo)):
                envi.save_image(
                    str(caminho), cubo, dtype=np.float32, interleave="bsq",
                    ext=".raw", metadata=metadados,
                )

            saida_tif = pasta / "saida" / "erro.tif"
            saida_png = pasta / "saida" / "erro.png"
            treinar_e_detectar_anomalias(
                treino_hdr, alvo_hdr, saida_tif, saida_png, epocas=3, semente=42,
            )

            self.assertTrue(saida_tif.is_file())
            self.assertTrue(saida_png.is_file())
            with rasterio.open(saida_tif) as resultado:
                self.assertEqual(resultado.count, 1)
                self.assertEqual(resultado.crs, rasterio.crs.CRS.from_epsg(32722))
                self.assertEqual(resultado.transform, from_origin(500000, 8000000, 30, 30))
                erro = resultado.read(1)
            self.assertEqual(erro.shape, (12, 12))
            self.assertTrue(np.isfinite(erro).all())
            self.assertTrue((erro >= 0).all())
            self.assertGreater(erro[anomalia].mean(), erro[~anomalia].mean())
            self.assertEqual(imread(saida_png).shape[:2], (12, 12))


if __name__ == "__main__":
    unittest.main()
