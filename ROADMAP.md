# ROADMAP — ANNA

**Status:** PRONTO PARA REVISÃO · Projeto de hackathon restaurado em 10/11/2026

## Checklist de "pronto"
- [x] Código organizado em pacote com linha de comando (`python -m anna`)
- [x] Testes com dados sintéticos (28; treino opcional com TensorFlow) e CI verde
- [x] README com contexto, diagrama do pipeline, resultados e como obter os dados da NASA
- [x] Dados brutos (12 GB) fora do repositório
- [x] Dependabot ativo (pip + GitHub Actions)
- [ ] Nomes da equipe e do hackathon no README (TODO para o dono)
- [ ] Licença definida com a equipe

## Backlog
1. Validar com polígonos de mineração conhecidos (MapBiomas): precisão e revocação por pixel.
2. Comparar o autoencoder com Isolation Forest e PCA.
3. Mapas de abundância mineral a partir das bandas SWIR.
4. [x] Teste de ponta a ponta do treino e da detecção com cubos ENVI sintéticos pequenos, três épocas
   e semente fixa; verifica TIF, PNG e erro maior na anomalia. Pula automaticamente sem TensorFlow.

## Log
- 2026-11-10 — Projeto encontrado em `OneDrive\Documentos\codecraft` (nome antigo); código original publicado na `main`;
  refatoração em pacote, testes, CI e README na `dev`.
- 2026-11-10 — Projeto renomeado de CodeCraft para ANNA (repositório, pacote `anna` e linha de comando `python -m anna`).
