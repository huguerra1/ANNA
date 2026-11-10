"""NDWI, máscara de água e contraste calculado apenas em terra."""

import unittest

import numpy as np

from anna.refinamento import calcular_ndwi, criar_mascara_agua, refinar_contraste


class TestNDWI(unittest.TestCase):
    def test_formula_e_limiar_estrito(self):
        verde = np.array([[3., 3., 1., 0.]])
        nir = np.array([[1., 2., 3., 0.]])
        ndwi = calcular_ndwi(verde, nir)
        np.testing.assert_allclose(ndwi, [[0.5, 0.2, -0.5, np.nan]], equal_nan=True)
        np.testing.assert_array_equal(criar_mascara_agua(ndwi), [[True, False, False, False]])

    def test_divisao_por_zero_preserva_semantica_sem_mudar_configuracao_global(self):
        configuracao = np.geterr().copy()
        ndwi = calcular_ndwi([0., 1.], [0., -1.])
        self.assertTrue(np.isnan(ndwi[0]))
        self.assertTrue(np.isposinf(ndwi[1]))
        self.assertEqual(np.geterr(), configuracao)

    def test_bandas_com_formatos_diferentes(self):
        with self.assertRaisesRegex(ValueError, "mesmo formato"):
            calcular_ndwi(np.ones((2, 2)), np.ones((2, 1)))


class TestRefinarContraste(unittest.TestCase):
    def test_agua_zerada_e_percentil_apenas_na_terra(self):
        mapa = np.array([[10000., 1., 2.], [20000., 3., 4.]])
        agua = np.array([[True, False, False], [True, False, False]])
        original = mapa.copy()
        vmax = np.percentile([1., 2., 3., 4.], 98)
        esperado = np.clip(np.where(agua, 0, mapa), 0, vmax) / vmax
        resultado = refinar_contraste(mapa, agua)
        np.testing.assert_allclose(resultado, esperado)
        np.testing.assert_array_equal(resultado[agua], 0)
        np.testing.assert_array_equal(mapa, original)

    def test_sem_terra_tem_erro_claro(self):
        with self.assertRaisesRegex(ValueError, "pixels de referência"):
            refinar_contraste(np.ones((2, 2)), np.ones((2, 2), dtype=bool))

    def test_mapa_e_mascara_com_formatos_diferentes(self):
        with self.assertRaisesRegex(ValueError, "mesmo formato"):
            refinar_contraste(np.ones((2, 2)), np.ones((2, 1), dtype=bool))


if __name__ == "__main__":
    unittest.main()
