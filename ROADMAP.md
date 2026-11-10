# ROADMAP — CodeCraft

**Status:** PRONTO PARA REVISÃO · Projeto de hackathon restaurado em 10/11/2026

## Checklist de "pronto"
- [x] Código organizado em pacote com linha de comando (`python -m codecraft`)
- [x] Testes com dados sintéticos (26) e CI verde
- [x] README com contexto, diagrama do pipeline, resultados e como obter os dados da NASA
- [x] Dados brutos (12 GB) fora do repositório
- [x] Dependabot ativo (pip + GitHub Actions)
- [ ] Nomes da equipe e do hackathon no README (TODO para o dono)
- [ ] Licença definida com a equipe

## Backlog
1. Validar com polígonos de mineração conhecidos (MapBiomas): precisão e revocação por pixel.
2. Comparar o autoencoder com Isolation Forest e PCA.
3. Mapas de abundância mineral a partir das bandas SWIR.
4. Teste do treino do autoencoder com um cubo sintético pequeno (hoje só a parte sem TensorFlow é testada).

## Log
- 2026-11-10 — Projeto encontrado em `OneDrive\Documentos\codecraft`; código original publicado na `main`;
  refatoração em pacote, testes, CI e README na `dev`.
