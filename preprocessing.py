import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

sns.set_theme(style="whitegrid")
TARGET = "Adaptivity Level"
ORDER = ["Low", "Moderate", "High"]

df = pd.read_excel("students_adaptability_level_online_education.xlsx",
                   sheet_name=0, header=1)
df.columns = df.columns.str.strip()
for c in df.columns:
    df[c] = df[c].astype(str).str.strip()

print(df.shape, "| missing:", df.isna().sum().sum(), "| duplikat:", df.duplicated().sum())
print(df[TARGET].value_counts())

# 1. EDA BIVARIAT (BAB II) - 100% stacked bar

def stacked_bar(ax, col):
    ct = pd.crosstab(df[col], df[TARGET], normalize="index")[ORDER] * 100
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#d9534f", "#f0ad4e", "#5cb85c"], width=0.8)
    ax.set_title(f"{col} vs {TARGET}")
    ax.set_ylabel("Persentase (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title=TARGET, loc="upper right", fontsize=8)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.ravel(), ["Internet Type", "Financial Condition",
                                  "Institution Type", "Class Duration"]):
    stacked_bar(ax, col)
plt.tight_layout()
plt.savefig("bivariat_stacked_bar.png", dpi=150)
plt.show()

# 2. UJI CHI-SQUARE & CRAMER'S V (BAB II)

def chi_square(a, b):
    table = pd.crosstab(a, b)
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return chi2, p, dof, v

# 2a. Setiap prediktor vs target

rows = []
for col in df.columns.drop(TARGET):
    chi2, p, dof, v = chi_square(df[col], df[TARGET])
    rows.append({"Atribut": col, "Chi2": round(chi2, 2), "dof": dof,
                 "p-value": p, "Cramer's V": round(v, 3)})
chi_target = pd.DataFrame(rows).sort_values("Cramer's V", ascending=False)
print("\nChi-Square: prediktor vs target\n", chi_target.to_string(index=False))

# 2b. Antar-prediktor (deteksi redundansi)

features = list(df.columns.drop(TARGET))
vmat = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
pairs = []
for i, a in enumerate(features):
    for b in features[i + 1:]:
        chi2, p, dof, v = chi_square(df[a], df[b])
        vmat.loc[a, b] = vmat.loc[b, a] = v
        pairs.append({"Atribut A": a, "Atribut B": b, "Chi2": round(chi2, 2),
                      "p-value": p, "Cramer's V": round(v, 3)})
chi_pairs = pd.DataFrame(pairs).sort_values("Cramer's V", ascending=False)
print("\n10 pasangan atribut dengan asosiasi tertinggi\n", chi_pairs.head(10).to_string(index=False))

plt.figure(figsize=(10, 8))
sns.heatmap(vmat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
plt.title("Heatmap Cramer's V antar-atribut")
plt.tight_layout()
plt.savefig("heatmap_cramers_v.png", dpi=150)
plt.show()

# 3. ENCODING (BAB III)

X = df.drop(columns=TARGET)
y = df[TARGET].map({"Low": 0, "Moderate": 1, "High": 2})

X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)
print("\nDimensi fitur setelah One-Hot:", X_encoded.shape)

# 4. SPLIT & SMOTE (PERBAIKAN DATA LEAKAGE)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y,         
)

print("\nSebelum SMOTE (train):", y_train.value_counts().sort_index().to_dict())
print("Test (tidak diubah)  :", y_test.value_counts().sort_index().to_dict())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Sesudah SMOTE (train):", y_train_res.value_counts().sort_index().to_dict())
print("Dimensi X_train_res  :", X_train_res.shape, "| X_test:", X_test.shape)

frac = ((X_train_res % 1) != 0).any(axis=1).sum()
print(f"Baris sintetis dengan nilai non-biner: {frac}")

# 5. TABEL SEBELUM vs SESUDAH ONE-HOT

ohe_summary = pd.DataFrame({
    "Atribut Asli": X.columns,
    "Jumlah Kategori": [X[c].nunique() for c in X.columns],
    "Kolom Dummy": [X[c].nunique() - 1 for c in X.columns],
    "Kategori Referensi (drop first)": [sorted(X[c].unique())[0] for c in X.columns],
})
print("\n", ohe_summary.to_string(index=False), "\nTotal kolom dummy:", ohe_summary["Kolom Dummy"].sum())