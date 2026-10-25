# Experimento: seleção de instâncias x seleção de atributos em árvores CART.
# Por conjunto, são 3 repetições de validação cruzada com 5 partições. Cada partição
# gera 50 árvores (5 frações de instâncias x 5 de atributos x 2 seletores).
# Uso: python code/02_experimento.py [nome ...]   (N_JOBS=4 muda o paralelismo)
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

NIVEIS = [0.2, 0.4, 0.6, 0.8, 1.0]  # frações retidas, iguais nos dois eixos
SEEDS = [0, 1, 2]                   # uma semente por repetição da validação cruzada
N_JOBS = int(os.environ.get("N_JOBS", "2"))
os.makedirs("results/raw", exist_ok=True)


def carrega(nome):
    # Pré-processamento feito uma vez por conjunto. Não usa o rótulo.
    df = pd.read_csv(f"datasets/{nome}.csv")
    y = pd.factorize(df["target"])[0]
    X = df.drop(columns="target")
    cols = []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            # numérico: falta vira a mediana (0 se a coluna for toda vazia)
            mediana = s.median() if s.notna().any() else 0
            cols.append(s.fillna(mediana).astype(float).to_numpy())
        else:
            # categórico: código ordinal; a falta vira -1 e conta como categoria própria
            cols.append(pd.factorize(s)[0].astype(float))
    return np.column_stack(cols), y


def roda_tarefa(nome, seed, fold):
    # Uma tarefa = um conjunto, uma repetição, uma partição. Cada uma grava um arquivo
    # em results/parts/, então dá para interromper e retomar sem perder o que já rodou.
    parte = f"results/parts/{nome}__s{seed}_f{fold}.csv"
    if os.path.exists(parte):
        return 0.0
    t0 = time.time()

    X, y = carrega(nome)
    n, d = X.shape
    tr, te = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))[fold]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    # Ranking de atributos por informação mútua, calculado só no treino completo.
    # Assim o teste não vaza para a seleção e o ranking não muda com as instâncias.
    ranking = np.argsort(-mutual_info_classif(Xtr, ytr, random_state=seed))
    rng = np.random.RandomState(1000 * seed + fold)

    linhas = []
    for fi in NIVEIS:
        if fi < 1.0:
            # subamostra aleatória que mantém a proporção das classes
            idx, _ = train_test_split(np.arange(len(ytr)), train_size=fi, stratify=ytr,
                                      random_state=int(rng.randint(1 << 30)))
        else:
            idx = np.arange(len(ytr))

        for fa in NIVEIS:
            k = max(1, math.ceil(fa * d))  # quantidade de atributos mantidos
            for seletor in ("mi", "rand"):
                # mi: os k mais informativos; rand: k sorteados (controle)
                cols = ranking[:k] if seletor == "mi" else rng.choice(d, k, replace=False)

                t1 = time.perf_counter()
                arvore = DecisionTreeClassifier(criterion="gini", random_state=seed)
                arvore.fit(Xtr[idx][:, cols], ytr[idx])
                tempo = time.perf_counter() - t1

                # o teste mantém todas as instâncias e usa só as colunas escolhidas
                pred = arvore.predict(Xte[:, cols])
                usados = arvore.tree_.feature[arvore.tree_.feature >= 0]  # nós internos

                linhas.append(dict(
                    dataset=nome, seed=seed, fold=fold, fi=fi, fa=fa, estrategia=seletor,
                    n_treino=len(idx), k_attr=k,
                    profundidade=arvore.get_depth(),
                    nos=arvore.tree_.node_count,
                    folhas=arvore.get_n_leaves(),
                    attr_usados=len(np.unique(usados)),
                    f1_macro=f1_score(yte, pred, average="macro"),
                    acc=accuracy_score(yte, pred),
                    t_treino=tempo,
                ))

    # grava em arquivo temporário e renomeia, para não deixar parte pela metade
    pd.DataFrame(linhas).to_csv(parte + ".tmp", index=False)
    os.replace(parte + ".tmp", parte)
    return time.time() - t0


def junta(nomes):
    # Quando as 15 partes de um conjunto existem, une tudo em results/raw/<nome>.csv.
    for nome in nomes:
        saida = f"results/raw/{nome}.csv"
        partes = [f"results/parts/{nome}__s{s}_f{f}.csv" for s in SEEDS for f in range(5)]
        if not os.path.exists(saida) and all(os.path.exists(p) for p in partes):
            pd.concat([pd.read_csv(p) for p in partes]).to_csv(saida, index=False)


if __name__ == "__main__":
    os.makedirs("results/parts", exist_ok=True)
    nomes = pd.read_csv("results/datasets_selecionados.csv")["name"].tolist()
    if len(sys.argv) > 1:
        nomes = sys.argv[1:]
    # os menores primeiro, para ver resultado cedo
    nomes = sorted(nomes, key=lambda n: os.path.getsize(f"datasets/{n}.csv"))

    tarefas = [
        (n, s, f)
        for n in nomes if not os.path.exists(f"results/raw/{n}.csv")
        for s in SEEDS for f in range(5)
        if not os.path.exists(f"results/parts/{n}__s{s}_f{f}.csv")
    ]
    print(len(tarefas), "tarefas pendentes", flush=True)

    resultados = Parallel(n_jobs=N_JOBS, return_as="generator")(
        delayed(roda_tarefa)(n, s, f) for n, s, f in tarefas
    )
    for (n, s, f), seg in zip(tarefas, resultados):
        print(f"{n} s{s} f{f}: {seg:.0f}s", flush=True)
        junta([n])

    junta(nomes)
    feitos = sum(os.path.exists(f"results/raw/{n}.csv") for n in nomes)
    print(f"conjuntos completos: {feitos}/{len(nomes)}", flush=True)
