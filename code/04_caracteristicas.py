# Análises exploratórias do artigo, a partir de results/resumo_celulas.csv:
# (1) efeito da redução em função de n, d e K (Spearman, Mann-Whitney, Fisher);
# (2) tabela de compromisso entre redução de nós e perda de F1 (Figura 3).
# Uso: python code/04_caracteristicas.py   (rode depois de 03_analise.py)
import pandas as pd
from scipy import stats

cat = pd.read_csv("results/datasets_selecionados.csv").set_index("name")
cel = pd.read_csv("results/resumo_celulas.csv")
mi = cel[cel.estrategia == "mi"]


def celula(fi, fa, coluna):
    # valor de uma coluna para cada conjunto, na célula (fi, fa)
    return mi[(mi.fi == fi) & (mi.fa == fa)].set_index("dataset")[coluna]


# Uma linha por conjunto: redução de nós e variação de F1 com 20% das instâncias (ri, fi_)
# e com 20% dos atributos (ra, fa_), mais as características do conjunto.
car = pd.DataFrame({
    "ri": celula(0.2, 1.0, "red_nos"), "fi_": celula(0.2, 1.0, "f1_rel"),
    "ra": celula(1.0, 0.2, "red_nos"), "fa_": celula(1.0, 0.2, "f1_rel"),
}).join(cat[["n", "d", "k"]])

print("Correlações de Spearman (rho, p)")
for x, y, rotulo in [("n", "ri", "n x redução de nós (20% instâncias)"),
                     ("d", "ri", "d x redução de nós (20% instâncias)"),
                     ("k", "fi_", "K x variação de F1 (20% instâncias)"),
                     ("d", "fa_", "d x variação de F1 (20% atributos)")]:
    rho, p = stats.spearmanr(car[x], car[y])
    print(f"  {rotulo}: {rho:+.2f}  p={p:.2g}")

# Conjuntos com muitas classes (K >= 6) contra os demais
k6, k2 = car[car.k >= 6], car[car.k < 6]
mw = stats.mannwhitneyu(k6.fi_, k2.fi_).pvalue
print(f"\nK>=6: {len(k6)} conjuntos, F1 mediano {k6.fi_.median():+.3f}; "
      f"demais: {k2.fi_.median():+.3f}; Mann-Whitney p={mw:.2g}")

# Conjuntos com poucos atributos (d <= 10): em quantos o número de nós aumentou com 20% dos atributos
pq, gr = car[car.d <= 10], car[car.d > 10]
aum_pq, aum_gr = int((pq.ra < 0).sum()), int((gr.ra < 0).sum())
fisher = stats.fisher_exact([[aum_pq, len(pq) - aum_pq], [aum_gr, len(gr) - aum_gr]]).pvalue
print(f"d<=10: aumento em {aum_pq} de {len(pq)}; demais: {aum_gr} de {len(gr)}; Fisher p={fisher:.3f}")

# Conjuntos sem configuração viável e seu número de classes
sem = pd.Series(__import__("json").load(open("results/analise.json"))["B"]["mi_principal"]["sem_viavel"])
print("\nSem configuração viável:", {n: int(cat.k[n]) for n in sem})

# Figura 3: mediana, entre os conjuntos, da redução de nós e da perda de F1 em cada configuração
tab = mi.groupby(["fi", "fa"]).agg(reducao_nos=("red_nos", "median"), f1_rel=("f1_rel", "median")).reset_index()
tab["perda_f1"] = -tab.f1_rel
print("\nCompromisso por configuração (medianas)")
print(tab[["fi", "fa", "reducao_nos", "perda_f1"]].round(3).to_string(index=False))
