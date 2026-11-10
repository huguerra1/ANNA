"""Testes dos cálculos espectrais, sem TensorFlow ou arquivos de satélite."""

import unittest

import numpy as np

from anna.processamento import (
    encontrar_banda_mais_proxima,
    mascara_pixels_validos,
    normalizar_por_percentil,
)
from anna.rgb import compor_rgb


class TestBandaMaisProxima(unittest.TestCase):
    def test_alvo_exato_aproximado_e_fora_do_intervalo(self):
        wavelengths = [450, 550, 650, 860]
        for alvo, esperado in [(550, 1), (640, 2), (100, 0), (1000, 3)]:
            with self.subTest(alvo=alvo):
                self.assertEqual(encontrar_banda_mais_proxima(wavelengths, alvo), esperado)

    def test_empate_escolhe_primeira_banda(self):
        self.assertEqual(encontrar_banda_mais_proxima([450, 550], 500), 0)

    def test_lista_vazia_tem_erro_claro(self):
        with self.assertRaisesRegex(ValueError, "não vazia"):
            encontrar_banda_mais_proxima([], 550)


class TestMascaraPixelsValidos(unittest.TestCase):
    def test_nodata_em_qualquer_banda_e_soma_zero_ou_negativa(self):
        pixels = np.array([
            [1, 2, 3],
            [-9999, 20000, 1],  # Mesmo com soma positiva, nodata invalida.
            [20000, -9999, 1],
            [20000, 1, -9999],
            [0, 0, 0],
            [-1, 1, 0],
            [-3, 1, 1],
            [-1, 2, 0],  # Valores negativos isolados não invalidam o espectro.
        ], dtype=float)
        np.testing.assert_array_equal(
            mascara_pixels_validos(pixels),
            [True, False, False, False, False, False, False, True],
        )


class TestNormalizacaoPorPercentil(unittest.TestCase):
    def test_percentis_globais_e_recorte(self):
        dados = np.arange(101, dtype=float).reshape(1, 101)
        resultado = normalizar_por_percentil(dados)
        esperado = (np.clip(dados, 2, 98) - 2) / 96
        np.testing.assert_allclose(resultado, esperado)
        self.assertEqual(resultado[0, 0], 0)
        self.assertEqual(resultado[0, -1], 1)

    def test_mapa_mantem_limite_inferior_zero(self):
        dados = np.array([2., 4., 8., 100.])
        esperado = np.clip(dados, 0, np.percentile(dados, 98)) / np.percentile(dados, 98)
        np.testing.assert_allclose(
            normalizar_por_percentil(dados, percentil_inferior=None), esperado
        )
        self.assertGreater(esperado[0], 0)

    def test_imagem_constante_e_mapa_sem_erro(self):
        np.testing.assert_array_equal(normalizar_por_percentil(np.ones((2, 3))), np.zeros((2, 3)))
        np.testing.assert_array_equal(
            normalizar_por_percentil(np.zeros((2, 3)), percentil_inferior=None),
            np.zeros((2, 3)),
        )

    def test_sem_referencia_tem_erro_claro(self):
        with self.assertRaisesRegex(ValueError, "pixels de referência"):
            normalizar_por_percentil(np.array([]))

    def test_composicao_rgb_preserva_formula_original_e_entrada(self):
        dados = np.array([[[-9999., 1., 2.], [3., 4., 100.]]], dtype=np.float32)
        original = dados.copy()
        sem_nodata = np.where(dados == -9999, 0, dados)
        p2, p98 = np.percentile(sem_nodata, (2, 98))
        esperado = ((np.clip(sem_nodata, p2, p98) - p2) / (p98 - p2) * 255).astype(np.uint8)
        resultado = compor_rgb(dados)
        self.assertEqual(resultado.dtype, np.uint8)
        np.testing.assert_array_equal(resultado, esperado)
        np.testing.assert_array_equal(dados, original)


if __name__ == "__main__":
    unittest.main()
