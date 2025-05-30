#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ——— CONFIGURATION —————————————————————————————————————————————
CSV_PATH = "model_summary_analysis.csv"

# Column names in your CSV
ERR_COL     = "mean_Force_Error"
THRPT_COL   = "mean_katom_steps_per_s_log"
TIMESTEP_COL = "mean_timesteps_per_s_log"
LABEL_COL   = "hpo_id"

# ——— 1. Load data —————————————————————————————————————————————————
df = pd.read_csv(CSV_PATH)
print(df.columns)

# ——— 2. Compute timesteps per second —————————————————————————————
# mean_Throughput is in k-atom-steps/s → multiply by 1e3 and divide by atom count
#df["timesteps_per_s"] = (df[THRPT_COL] * 1e3) / df[ATOMS_COL]

# ——— 3. Plot histogram of Force RMSE —————————————————————————————
mean_err = df[ERR_COL].mean()

plt.figure(figsize=(6,4))
plt.hist(df[ERR_COL], bins=30, alpha=0.7, edgecolor="k")
plt.axvline(mean_err, color="r", linestyle="--", linewidth=2,
            label=f"Mean = {mean_err:.4f} Å")
plt.xlabel("Force RMSE (Å)")
plt.ylabel("Count")
plt.title("Histogram of Force Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Show or save
plt.show()


# ——— 4. Subset to low-error models (below mean) —————————————————————
df_low = df[df[ERR_COL] < mean_err].reset_index(drop=True)

# ——— 5. Compute 2-D Pareto front (error ↓ vs timesteps/s ↑) —————————
errs  = df_low[ERR_COL].values
times = df_low[TIMESTEP_COL].values

is_pareto = np.ones(len(df_low), dtype=bool)
for i, (e_i, t_i) in enumerate(zip(errs, times)):
    # if any other model is at least as good in both dims
    # and strictly better in one, then i is not Pareto
    if np.any((errs <= e_i) & (times >= t_i) & ((errs < e_i) | (times > t_i))):
        is_pareto[i] = False

df_pareto = df_low[is_pareto].sort_values(ERR_COL).reset_index(drop=True)

# ——— 6. Plot Pareto front & annotate —————————————————————————————
plt.figure(figsize=(8,6))
plt.scatter(df_low[ERR_COL], df_low[TIMESTEP_COL],
            s=30, alpha=0.4, label="Low-error models")

plt.scatter(df_pareto[ERR_COL], df_pareto[TIMESTEP_COL],
            s=100, edgecolor="k", facecolor="none", linewidth=1.5,
            label="Pareto frontier")

# annotate each Pareto point with its HPO ID
for _, row in df_pareto.iterrows():
    plt.annotate(
        row[LABEL_COL],
        xy=(row[ERR_COL], row[TIMESTEP_COL]),
        xytext=(5, -3),
        textcoords="offset points",
        fontsize=8,
        color="k"
    )

plt.xlabel("Force RMSE (Å)")
plt.ylabel("Timesteps per second")
plt.title("2D Pareto: Error ↓ vs Timesteps/s ↑ (Error < Mean)")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Show or save
plt.show()
