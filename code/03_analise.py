# Análise estatística: lê results/raw/*.csv e gera results/analise.json e resumo_celulas.csv.
import glob
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

NIV = [0.2, 0.4, 0.6, 0.8, 1.0]
METRICAS = ["profundidade", "nos", "folhas", "attr_usados", "f1_macro", "acc", "t_treino"]

raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/raw/*.csv"))])
cat = pd.read_csv("results/datasets_selecionados.csv").set_index("name")

# Média das 15 partições, por conjunto e por célula (seletor, fi, fa)
cell = raw.groupby(["dataset", "estrategia", "fi", "fa"])[METRICAS].mean().reset_index()

# Linha de base: (fi, fa) = (100%, 100%) de cada conjunto e seletor
base = cell[(cell.fi == 1.0) & (cell.fa == 1.0)].set_index(["dataset", "estrategia"])
for m in ["profundidade", "nos", "folhas", "f1_macro"]:
    cell[m + "_b"] = [base.loc[(d, e), m] for d, e in zip(cell.dataset, cell.estrategia)]

# Variações em relação à linha de base
cell["red_nos"] = 1 - cell.nos / cell.nos_b                          # >0 reduz, <0 aumenta
cell["red_prof"] = 1 - cell.profundidade / cell.profundidade_b
cell["f1_rel"] = (cell.f1_macro - cell.f1_macro_b) / cell.f1_macro_b
cell["f1_pts"] = cell.f1_macro - cell.f1_macro_b

nds = cell.dataset.nunique()
out = {"n_datasets": int(nds), "n_arvores": int(len(raw))}


def holm(p):
    # Correção de Holm: ordena os p, multiplica por (m - posição) e garante monotonia
    p = np.asarray(p)
    ordem = np.argsort(p)
    m = len(p)
    ajust = np.empty(m)
    maximo = 0
    for pos, i in enumerate(ordem):
        maximo = max(maximo, (m - pos) * p[i])
        ajust[i] = min(1, maximo)
    return ajust


# H1a: reduzir só atributos x reduzir só instâncias, na mesma fração retida.
# Wilcoxon pareado entre os conjuntos, com Holm sobre os 4 níveis.
tabA = {}
for estr in ("mi", "rand"):
    for metrica, nome in (("red_nos", "nos"), ("red_prof", "prof")):
        linhas, ps = [], []
        for L in NIV[:-1]:
            a = cell[(cell.estrategia == estr) & (cell.fi == 1.0) & (cell.fa == L)].set_index("dataset")[metrica]
            i = cell[(cell.estrategia == estr) & (cell.fi == L) & (cell.fa == 1.0)].set_index("dataset")[metrica]
            i = i.reindex(a.index)
            p_bi = stats.wilcoxon(a, i, alternative="two-sided").pvalue
            p_maior = stats.wilcoxon(a, i, alternative="greater").pvalue  # sentido de H1a: atributos reduzem mais
            linhas.append(dict(
                L=L, med_attr=a.median(), med_inst=i.median(),
                p_two=p_bi, p_greater=p_maior,
                n_attr_aumenta=int((a < 0).sum()),
                n_inst_aumenta=int((i < 0).sum()),
                attr_maior=int((a > i).sum()),
            ))
            ps.append(p_bi)
        for linha, q in zip(linhas, holm(ps)):
            linha["p_two_holm"] = q
        tabA[f"{estr}_{nome}"] = linhas
out["A"] = tabA


# H1b: existe configuração viável? (corta >= 30% dos nós e perde <= 5% de F1 relativo)
def tipo(r):
    if r.fa == 1.0:
        return "instancias"
    if r.fi == 1.0:
        return "atributos"
    return "conjunto"


B = {}
for estr in ("mi", "rand"):
    c = cell[(cell.estrategia == estr) & ~((cell.fi == 1.0) & (cell.fa == 1.0))].copy()
    c["tipo"] = c.apply(tipo, axis=1)

    # critério principal e dois de sensibilidade: (corte mínimo de nós, perda máxima de F1)
    for corte, perda, tag in ((0.30, -0.05, "principal"), (0.30, -0.02, "estrito"), (0.50, -0.05, "agressivo")):
        ok = c[(c.red_nos >= corte) & (c.f1_rel >= perda)]
        # melhor configuração de cada conjunto = a que mais reduz nós
        melhor = ok.loc[ok.groupby("dataset").red_nos.idxmax()] if len(ok) else ok
        k = melhor.dataset.nunique() if len(ok) else 0
        ic = stats.binomtest(k, nds).proportion_ci(method="exact")  # Clopper-Pearson
        B[f"{estr}_{tag}"] = dict(
            n_viaveis=int(k), frac=k / nds, ci_lo=ic.low, ci_hi=ic.high,
            tipo_melhor=melhor.tipo.value_counts().to_dict() if len(ok) else {},
            med_red_nos=float(melhor.red_nos.median()) if len(ok) else None,
            med_f1_rel=float(melhor.f1_rel.median()) if len(ok) else None,
            med_fi=float(melhor.fi.median()) if len(ok) else None,
            med_fa=float(melhor.fa.median()) if len(ok) else None,
            sem_viavel=sorted(set(c.dataset) - set(ok.dataset)),
        )

    # quantos conjuntos têm alguma configuração viável de cada tipo
    ok = c[(c.red_nos >= .30) & (c.f1_rel >= -.05)]
    B[f"{estr}_por_tipo"] = {
        t: int(ok[ok.tipo == t].dataset.nunique()) for t in ("instancias", "atributos", "conjunto")
    }
out["B"] = B

# Medianas por célula, usadas nos mapas de calor da Figura 2
H = {}
for estr in ("mi", "rand"):
    g = cell[cell.estrategia == estr].groupby(["fi", "fa"])
    H[estr] = {
        m: g[m].median().unstack("fa").reindex(index=NIV, columns=NIV).values.tolist()
        for m in ("red_nos", "red_prof", "f1_rel", "f1_pts")
    }
    H[estr]["iqr_nos"] = {}
    for fi in NIV:
        for fa in NIV:
            x = cell[(cell.estrategia == estr) & (cell.fi == fi) & (cell.fa == fa)].red_nos
            H[estr]["iqr_nos"][f"{fi}_{fa}"] = [float(v) for v in np.percentile(x, [25, 75])]
out["H"] = H

# Linha de base (seletor mi): tamanho da árvore x tamanho do problema (Spearman)
b = base.xs("mi", level="estrategia").join(cat[["n", "d", "k"]])
D = {}
for x in ("d", "n", "k"):
    for y in ("profundidade", "nos"):
        rho, p = stats.spearmanr(b[x], b[y])
        D[f"{x}_{y}"] = dict(rho=float(rho), p=float(p))
out["D"] = D

# Friedman: efeito dos 5 níveis de cada eixo sobre nós e F1
F = {}
for eixo in ("fi", "fa"):
    for metrica in ("nos", "f1_macro"):
        outro = (cell.fa == 1.0) if eixo == "fi" else (cell.fi == 1.0)
        piv = cell[(cell.estrategia == "mi") & outro].pivot(index="dataset", columns=eixo, values=metrica)
        chi2, p = stats.friedmanchisquare(*[piv[c] for c in NIV])
        F[f"mi_{eixo}_{metrica}"] = dict(chi2=float(chi2), p=float(p))
out["F"] = F

# Resumo da linha de base e das características dos conjuntos
out["baseline"] = dict(
    frac_attr_usados_med=float((b.attr_usados / b.d).median()),
    f1_med=float(b.f1_macro.median()),
    nos_med=float(b.nos.median()),
    prof_med=float(b.profundidade.median()),
    nos_min=float(b.nos.min()),
    nos_max=float(b.nos.max()),
)
out["dataset_stats"] = dict(
    n_min=int(cat.n.min()), n_max=int(cat.n.max()),
    d_min=int(cat.d.min()), d_max=int(cat.d.max()),
    k_min=int(cat.k.min()), k_max=int(cat.k.max()),
    n_med=float(cat.n.median()), d_med=float(cat.d.median()),
)

cell.to_csv("results/resumo_celulas.csv", index=False)
json.dump(out, open("results/analise.json", "w"), indent=1, default=float)
print("conjuntos:", nds, "| árvores:", len(raw))
print("viáveis (mi, critério principal):", B["mi_principal"]["n_viaveis"], "de", nds)
