# Seleção de instâncias e de atributos em árvores de decisão

Código e resultados do artigo "Efeito da Seleção de Instâncias e de Atributos sobre a Estrutura de Árvores de Decisão" (Aprendizado de Máquina, UTFPR, 2026).

O experimento treina árvores CART (scikit-learn, Gini, sem poda) em 48 conjuntos do OpenML-CC18. Cada árvore usa uma fração das instâncias e uma fração dos atributos (20% a 100% em cada eixo). Os atributos são escolhidos por informação mútua, com um controle de atributos sorteados. A validação é cruzada estratificada, 5 partições com 3 repetições. Ao todo são 36.000 árvores.

## Como reproduzir

```bash
pip install -r requirements.txt
python code/baixar_cc18.py       # baixa os conjuntos para datasets/ (não versionado)
python code/01_prepara.py        # lista dos conjuntos que passam nos filtros
python code/02_experimento.py    # treina as árvores, grava em results/raw/
python code/03_analise.py        # testes estatísticos, grava results/analise.json
```

Rode os comandos a partir da raiz do repositório. O passo 3 pode ser interrompido e retomado, porque cada partição é salva à parte. A variável `N_JOBS` controla quantos processos rodam em paralelo.

## Pastas

- `code/`: os quatro scripts.
- `results/raw/`: uma tabela por conjunto, com uma linha por árvore (profundidade, nós, folhas, atributos usados, F1 macro).
- `results/`: catálogo dos conjuntos, lista dos 48 selecionados e as estatísticas finais.

Os dados originais são do OpenML e não estão neste repositório.
