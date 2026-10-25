# Baixa do OpenML os conjuntos escolhidos do CC18 e salva em datasets/<nome>.csv.
# A coluna alvo é renomeada para "target". Também gera results/catalogo_cc18.csv.
# Uso: python code/baixar_cc18.py
import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "datasets"
RES = RAIZ / "results"
DADOS.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# id do OpenML -> nome do conjunto
IDS = {
    3: "kr-vs-kp", 6: "letter", 11: "balance-scale", 12: "mfeat-factors",
    15: "breast-w", 23: "cmc", 28: "optdigits", 29: "credit-approval",
    31: "credit-g", 32: "pendigits", 37: "diabetes", 38: "sick",
    44: "spambase", 46: "splice", 50: "tic-tac-toe", 54: "vehicle",
    151: "electricity", 182: "satimage", 188: "eucalyptus", 307: "vowel",
    458: "analcatdata_authorship", 469: "analcatdata_dmft", 1053: "jm1",
    1461: "bank-marketing", 1462: "banknote-authentication",
    1464: "blood-transfusion", 1475: "first-order-theorem-proving",
    1480: "ilpd", 1486: "nomao", 1487: "ozone-level-8hr", 1489: "phoneme",
    1494: "qsar-biodeg", 1497: "wall-robot-navigation", 1501: "semeion",
    1510: "wdbc", 1590: "adult", 4534: "PhishingWebsites",
    4538: "GesturePhaseSegmentation", 6332: "cylinder-bands", 40499: "texture",
    40670: "dna", 40701: "churn", 40966: "MiceProtein", 40975: "car",
    40982: "steel-plates-fault", 40983: "wilt", 40984: "segment",
    41027: "jungle_chess_2pcs",
}

linhas = []
for did, nome in IDS.items():
    destino = DADOS / f"{nome}.csv"

    if destino.exists():
        # já baixado antes, só lê para atualizar o catálogo
        df = pd.read_csv(destino)
        print(f"[ok  ] {did:>6} {nome} (já baixado)")
    else:
        # até 3 tentativas, porque o OpenML às vezes falha
        for tentativa in range(3):
            try:
                b = fetch_openml(data_id=did, as_frame=True, parser="auto", cache=True)
                df = b.frame.copy()
                alvo = b.target_names[0] if b.target_names else df.columns[-1]
                df = df.rename(columns={alvo: "target"})
                # deixa o alvo na última coluna
                df = df[[c for c in df.columns if c != "target"] + ["target"]]
                df.to_csv(destino, index=False)
                print(f"[novo] {did:>6} {nome}  {df.shape}")
                break
            except Exception as e:
                print(f"[erro] {did} {nome} tentativa {tentativa + 1}: {e}")
                time.sleep(3)
        else:
            linhas.append(dict(id=did, nome=nome, status="FALHOU"))
            continue

    # contagem por classe, usada para o filtro de classe mínima
    cont = df["target"].value_counts()
    d = df.shape[1] - 1
    passa = bool(500 <= len(df) <= 50000 and d <= 300 and cont.min() >= 20)
    linhas.append(dict(
        id=did, nome=nome, n=len(df), d=d, classes=len(cont),
        classe_min=int(cont.min()), faltantes=int(df.isna().sum().sum()),
        passa_filtro=passa, status="ok",
    ))

cat = pd.DataFrame(linhas)
cat.to_csv(RES / "catalogo_cc18.csv", index=False)
print("\n", cat.to_string(index=False))
print("\nNão passam no filtro:", cat.loc[cat.get("passa_filtro", True) == False, "nome"].tolist())
print("Falhas de download:", cat.loc[cat["status"] == "FALHOU", "nome"].tolist())
