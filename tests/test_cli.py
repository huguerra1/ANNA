"""Contrato da CLI e importação independente do TensorFlow."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import unittest

from anna.__main__ import criar_parser, main
from tests.temporarios import diretorio_temporario


class TestCLI(unittest.TestCase):
    def test_ajuda_anomalias_mostra_opcoes_do_treino(self):
        resultado = subprocess.run(
            [sys.executable, "-m", "anna", "anomalias", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        self.assertIn("--epocas", resultado.stdout)
        self.assertIn("--semente", resultado.stdout)

    def test_padroes_do_treino(self):
        args = criar_parser().parse_args(["anomalias", "treino.hdr", "analise.hdr", "erro.tif", "erro.png"])
        self.assertEqual(args.epocas, 50)
        self.assertEqual(args.semente, 42)

    def test_parametros_do_treino(self):
        args = criar_parser().parse_args([
            "anomalias", "treino.hdr", "analise.hdr", "erro.tif", "erro.png",
            "--epocas", "3", "--semente", "7",
        ])
        self.assertEqual(args.epocas, 3)
        self.assertEqual(args.semente, 7)

    def test_entrada_inexistente_retorna_um_e_mensagem_sem_traceback(self):
        with diretorio_temporario() as temporario:
            ausente = str(Path(temporario) / "ausente.nc")
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                codigo = main(["converter", ausente, str(Path(temporario) / "saida")])
        self.assertEqual(codigo, 1)
        self.assertIn("Arquivo de entrada não encontrado", stderr.getvalue())
        self.assertIn(ausente, stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_modulos_e_ajuda_sem_tensorflow(self):
        # Um processo limpo bloqueia qualquer tentativa de importar TensorFlow.
        programa = """
import importlib.abc
import sys
class BloquearTensorFlow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'tensorflow':
            raise AssertionError('TensorFlow não deve ser importado neste teste')
sys.meta_path.insert(0, BloquearTensorFlow())
import anna.anomalias
import anna.conversao
import anna.rgb
import anna.refinamento
from anna.__main__ import main
main(['--help'])
"""
        resultado = subprocess.run(
            [sys.executable, "-c", programa], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        for comando in ("converter", "rgb", "anomalias", "refinar"):
            self.assertIn(comando, resultado.stdout)


if __name__ == "__main__":
    unittest.main()
