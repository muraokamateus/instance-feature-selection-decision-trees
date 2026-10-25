# Lê o catálogo e guarda só os conjuntos que passaram nos filtros de tamanho.
import pandas as pd

cat = pd.read_csv("results/catalogo_cc18.csv")
ok = cat[(cat.status == "ok") & (cat.passa_filtro)]
ok = ok.rename(columns={"nome": "name", "classes": "k"})
ok = ok[["name", "id", "n", "d", "k", "classe_min", "faltantes"]]
ok.to_csv("results/datasets_selecionados.csv", index=False)
print(len(ok), "conjuntos selecionados")
