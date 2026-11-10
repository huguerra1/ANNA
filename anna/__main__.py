"""Interface de linha de comando: python -m anna."""

import argparse
import sys


def inteiro_positivo(valor):
    valor = int(valor)
    if valor <= 0:
        raise argparse.ArgumentTypeError("O número de épocas deve ser positivo.")
    return valor


def semente_valida(valor):
    valor = int(valor)
    if not 0 <= valor < 2**32:
        raise argparse.ArgumentTypeError("A semente deve estar entre 0 e 4294967295.")
    return valor


def criar_parser():
    parser = argparse.ArgumentParser(
        prog="python -m anna", description="ANNA: detecção de anomalias ambientais em imagens NASA EMIT."
    )
    comandos = parser.add_subparsers(dest="comando", required=True)

    converter = comandos.add_parser("converter", help="Converter NetCDF EMIT L2A para ENVI BSQ.")
    converter.add_argument("entrada_nc", help="Arquivo NetCDF de reflectância.")
    converter.add_argument("saida_base", help="Caminho de saída sem extensão (.hdr e .raw).")

    rgb = comandos.add_parser("rgb", help="Gerar composição RGB de uma imagem ENVI.")
    rgb.add_argument("entrada_hdr", help="Cabeçalho da imagem ENVI.")
    rgb.add_argument("saida_png", help="Imagem RGB de saída.")

    anomalias = comandos.add_parser("anomalias", help="Treinar autoencoder e detectar anomalias.")
    anomalias.add_argument("treino_hdr", help="Cabeçalho ENVI da área preservada.")
    anomalias.add_argument("analise_hdr", help="Cabeçalho ENVI da área de análise.")
    anomalias.add_argument("saida_tif", help="GeoTIFF com o erro real de reconstrução.")
    anomalias.add_argument("saida_png", help="Visualização com contraste acentuado.")
    anomalias.add_argument("--epocas", type=inteiro_positivo, default=50, help="Épocas de treino (padrão: 50).")
    anomalias.add_argument("--semente", type=semente_valida, default=42, help="Semente aleatória (padrão: 42).")

    refinar = comandos.add_parser("refinar", help="Mascarar água por NDWI e refinar o contraste.")
    refinar.add_argument("original_hdr", help="Cabeçalho ENVI da área de análise.")
    refinar.add_argument("anomalia_tif", help="GeoTIFF do erro de reconstrução.")
    refinar.add_argument("saida_png", help="Visualização refinada de saída.")
    return parser


def main(argv=None):
    args = criar_parser().parse_args(argv)
    # Importar apenas o comando solicitado mantém a CLI leve e o treino opcional.
    try:
        if args.comando == "converter":
            from .conversao import converter_emit_para_envi
            converter_emit_para_envi(args.entrada_nc, args.saida_base)
        elif args.comando == "rgb":
            from .rgb import converter_raw_para_rgb
            converter_raw_para_rgb(args.entrada_hdr, args.saida_png)
        elif args.comando == "anomalias":
            from .anomalias import treinar_e_detectar_anomalias
            treinar_e_detectar_anomalias(
                args.treino_hdr, args.analise_hdr, args.saida_tif, args.saida_png,
                epocas=args.epocas, semente=args.semente,
            )
        elif args.comando == "refinar":
            from .refinamento import refinar_mapa_anomalia
            refinar_mapa_anomalia(args.original_hdr, args.anomalia_tif, args.saida_png)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
