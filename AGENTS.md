# Regras do projeto CodeCraft

Instruções para agentes de código (Claude Code, Codex, Gemini, Kiro). `CLAUDE.md` aponta para este arquivo.

## O projeto
Detecção de anomalias ambientais em imagens hiperespectrais NASA EMIT com um autoencoder treinado numa área
preservada. Veja o README.

## Comandos
- `python -m unittest discover -s tests -v`: testes (não precisam do TensorFlow).
- `python -m codecraft --help`: linha de comando.

## Regras
1. Não altere os algoritmos nem os parâmetros que geraram os resultados do README sem registrar a mudança.
2. Dados brutos (`*.nc`, `*.raw`, `*.hdr`, `*.tif`) nunca entram no git.
3. Funções puras em `processamento.py`, com teste; entrada e saída nos demais módulos.
4. O TensorFlow só é importado dentro do treino.
5. Comentários e mensagens em português; commits curtos no imperativo.

## Autonomia
Trabalho em `claude/<tema>`, PR para a `dev` com CI verde. A `main` só muda quando o dono aprovar `dev → main`.
Nunca: apagar o repositório, mudar a visibilidade, commitar dados ou segredos.
