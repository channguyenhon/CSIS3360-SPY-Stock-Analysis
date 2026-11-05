"""Duc's retrospective SPY K-means comparison for Weeks 1 and 2.

Run from any directory. Paths resolve from this project's folder, not the cwd.
Compare K=2–6. Final K selection and market-profile naming belong to Week 3.
The scaler and model describe the full 2015–2025 period. Do not reuse them
for the classification model's preprocessing or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
STUDY_START = "2015-01-01"
STUDY_END = "2025-12-31"
FEATURES = ["Daily_Return", "Price_to_MA20_Ratio", "Volatility_20D", "Volume_Change"]
K_VALUES = [2, 3, 4, 5, 6]
KMEANS_SETTINGS = dict(init="k-means++", n_init=20,
                       random_state=42, max_iter=300, tol=1e-4, algorithm="lloyd")
COLORS = ["#2463A6", "#E18A2C", "#317B65", "#9962A7", "#C45050", "#71939A"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prices(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    expected = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if list(df.columns) != expected:
        raise ValueError(f"Expected columns {expected}; found {list(df.columns)}")
    dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError("Missing Date values")
    df["Date"] = dates
    for col in expected[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    values = df[expected[1:]].to_numpy(dtype=float)
    checks = {
        "missing_cells": int(df.isna().sum().sum()),
        "nonfinite_numeric_cells": int((~np.isfinite(values)).sum()),
        "duplicate_dates": int(df["Date"].duplicated().sum()),
        "nonpositive_prices_or_volume": int((values <= 0).sum()),
        "invalid_ohlc_rows": int(((df.High < df.Low) | (df.High < df.Open)
                                 | (df.High < df.Close) | (df.Low > df.Open)
                                 | (df.Low > df.Close)).sum()),
        "noninteger_volume_rows": int((df.Volume % 1 != 0).sum()),
        "weekend_rows": int((df.Date.dt.dayofweek >= 5).sum()),
    }
    if any(checks.values()):
        raise ValueError(f"Source data failed validation: {checks}. Review; do not silently repair.")
    sorted_input = bool(df.Date.is_monotonic_increasing)
    df = df.sort_values("Date").reset_index(drop=True)
    study = df.loc[df.Date.between(STUDY_START, STUDY_END)].copy().reset_index(drop=True)
    if len(study) < 22:
        raise ValueError("At least 22 study-period price observations are required")
    audit = {
        "source_file": path.name, "source_sha256": file_hash(path),
        "raw_rows": len(df), "raw_start": df.Date.min().strftime("%Y-%m-%d"),
        "raw_end": df.Date.max().strftime("%Y-%m-%d"),
        "study_rows": len(study), "rows_outside_study_period": len(df)-len(study),
        "study_start": study.Date.min().strftime("%Y-%m-%d"),
        "study_end": study.Date.max().strftime("%Y-%m-%d"),
        "source_already_chronological": sorted_input, "validation_counts": checks,
        "value_corrections": 0,
        "calendar_check": "Weekend dates checked; exchange holiday/calendar completeness not independently verified. No dates filled.",
        "price_basis": "Close as supplied; source adjustment methodology not verified.",
        "boundary_policy": "Filter to 2015–2025 first, then compute trailing windows; no pre-2015 warm-up.",
        "extremes_policy": "All valid observations retained; no clipping, winsorization, interpolation or backfill.",
    }
    return study, audit


def construct_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Every feature at t uses t and earlier observations only."""
    out = prices.copy()
    close, volume = out.Close, out.Volume
    out["Daily_Return"] = close / close.shift(1) - 1
    out["Previous_Day_Return"] = out.Daily_Return.shift(1)
    out["Return_5D"] = close / close.shift(5) - 1
    out["Return_20D"] = close / close.shift(20) - 1
    out["MA5"] = close.rolling(5, min_periods=5).mean()
    out["MA20"] = close.rolling(20, min_periods=20).mean()
    out["Price_to_MA5_Ratio"] = close / out.MA5
    out["Price_to_MA20_Ratio"] = close / out.MA20
    out["Relative_Price_Range"] = (out.High - out.Low) / close
    out["Volume_Change"] = volume / volume.shift(1) - 1
    out["Volume_Ratio_20D"] = volume / volume.rolling(20, min_periods=20).mean()
    out["Volatility_5D"] = out.Daily_Return.rolling(5, min_periods=5).std(ddof=1)
    out["Volatility_20D"] = out.Daily_Return.rolling(20, min_periods=20).std(ddof=1)
    # Shared targets are outputs for teammates, NEVER clustering inputs.
    next_close = close.shift(-1)
    out["Next_Trading_Date"] = out.Date.shift(-1)
    out["Next_Day_Return"] = next_close / close - 1
    out["Next_Day_Direction"] = (next_close > close).where(next_close.notna()).astype("Int64")
    # Use target date at period boundaries, not the feature date.
    split = pd.Series(pd.NA, index=out.index, dtype="string")
    split.loc[out.Next_Trading_Date.between("2015-01-01", "2022-12-31")] = "Train"
    split.loc[out.Next_Trading_Date.between("2023-01-01", "2023-12-31")] = "Validation"
    split.loc[out.Next_Trading_Date.between("2024-01-01", "2025-12-31")] = "Test"
    out["Predictive_Split"] = split
    return out


def variable_dictionary() -> pd.DataFrame:
    rows = [
        ("Daily_Return", "Close[t] / Close[t-1] - 1", "fraction", 1, True, "Current price direction"),
        ("Previous_Day_Return", "Daily_Return[t-1]", "fraction", 2, False, "Shared candidate for teammates; not in initial distance"),
        ("Return_5D", "Close[t] / Close[t-5] - 1", "fraction", 5, False, "Candidate momentum feature; not in initial distance"),
        ("Return_20D", "Close[t] / Close[t-20] - 1", "fraction", 20, False, "Candidate momentum feature; overlaps trend information"),
        ("MA5", "mean(Close[t-4:t]) inclusive", "same price units as Close", 4, False, "Intermediate level; not clustered"),
        ("MA20", "mean(Close[t-19:t]) inclusive", "same price units as Close", 19, False, "Intermediate level; not clustered"),
        ("Price_to_MA5_Ratio", "Close[t] / MA5[t]", "ratio, 1 means at MA5", 4, False, "Additional candidate; not in initial distance"),
        ("Price_to_MA20_Ratio", "Close[t] / MA20[t]", "ratio, 1 means at MA20", 19, True, "Relative position to a trailing trend"),
        ("Relative_Price_Range", "(High[t] - Low[t]) / Close[t]", "fraction", 0, False, "Candidate intraday risk feature; not in initial distance"),
        ("Volume_Change", "Volume[t] / Volume[t-1] - 1", "fraction", 1, True, "Change in market activity"),
        ("Volume_Ratio_20D", "Volume[t] / mean(Volume[t-19:t]) inclusive", "ratio", 19, False, "Additional activity candidate; not in initial distance"),
        ("Volatility_5D", "sample_std(Daily_Return[t-4:t]), ddof=1", "daily-return fraction; not annualized", 5, False, "Short volatility candidate; not in initial distance"),
        ("Volatility_20D", "sample_std(Daily_Return[t-19:t]), ddof=1", "daily-return fraction; not annualized", 20, True, "Trailing risk level; requires 21 Close observations"),
        ("Next_Trading_Date", "Date[t+1]", "date", 0, False, "Future metadata for target alignment; NEVER cluster input"),
        ("Next_Day_Return", "Close[t+1] / Close[t] - 1", "fraction", 0, False, "Regression target; NEVER cluster input; last row missing"),
        ("Next_Day_Direction", "1 if Close[t+1] > Close[t], else 0; NA if next Close unknown", "binary or missing", 0, False, "Classification target; NEVER cluster input"),
        ("Predictive_Split", "Train if target date in 2015–2022; Validation if 2023; Test if 2024–2025", "category", 0, False, "Predictive split metadata; NEVER cluster input"),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Definition", "Unit", "Warmup_rows", "Used_in_initial_KMeans", "Purpose"])


def common_modeling_table(features: pd.DataFrame) -> pd.DataFrame:
    # Fundamental plan requires one common table with complete rolling features and labels.
    return features.dropna().copy().reset_index(drop=True)


def fit_grid(eligible: pd.DataFrame):
    x = eligible[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x.std(axis=0) == 0).any():
        raise ValueError("Nonfinite or constant clustering feature")
    scaler = StandardScaler(with_mean=True, with_std=True)
    z = scaler.fit_transform(x)
    standardized = pd.DataFrame(z, columns=FEATURES)
    standardized.insert(0, "Date", eligible.Date)
    assignments = eligible[["Date"]].copy()
    models, comparison, sizes, centroid_rows = {}, [], [], []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, **KMEANS_SETTINGS)
        with threadpool_limits(limits=1):
            labels = model.fit_predict(z)
            score = float(silhouette_score(z, labels, metric="euclidean"))
        counts = pd.Series(labels).value_counts().sort_index()
        assignments[f"Cluster_K{k}"] = labels
        comparison.append({"K":k,"Silhouette":score,"Inertia_SSE":float(model.inertia_),
                           "Iterations":int(model.n_iter_),"Min_cluster_size":int(counts.min()),
                           "Max_cluster_size":int(counts.max()),"Observations":len(eligible)})
        for label,count in counts.items():
            sizes.append({"K":k,"Cluster":int(label),"Observations":int(count),"Share":float(count/len(eligible))})
        for label, center in enumerate(model.cluster_centers_):
            centroid_rows.append({"K":k,"Cluster":label,**dict(zip(FEATURES,center))})
        models[k] = model
    return standardized, assignments, scaler, models, pd.DataFrame(comparison), pd.DataFrame(sizes), pd.DataFrame(centroid_rows)


def create_figures(comparison: pd.DataFrame, sizes: pd.DataFrame, folder: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1,2,figsize=(8.0,3.6), layout="constrained")
    ax=axes[0]
    ax.plot(comparison.K,comparison.Silhouette,marker="o",color=COLORS[0],linewidth=2)
    for row in comparison.itertuples():
        ax.annotate(f"{row.Silhouette:.3f}",(row.K,row.Silhouette),xytext=(0,9),textcoords="offset points",ha="center",fontsize=9)
    ax.set(title="Silhouette score",xlabel="Number of clusters K",ylabel="Mean silhouette")
    ax.set_ylim(0,max(comparison.Silhouette)*1.20)
    ax=axes[1]
    ax.plot(comparison.K,comparison.Inertia_SSE,marker="o",color=COLORS[2],linewidth=2)
    ax.set(title="Within-cluster squared distance",xlabel="Number of clusters K",ylabel="Inertia / SSE (standardized units)")
    for ax in axes:
        ax.set_xticks(K_VALUES);ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("SPY K-means comparison for Week 2",fontsize=12,fontweight="bold")
    fig.savefig(folder/"k_comparison.png",dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(9.0,3.3),sharey=True,layout="constrained")
    for ax,k in zip(axes,K_VALUES):
        part=sizes.loc[sizes.K==k]
        bars=ax.bar(part.Cluster,part.Observations,color=COLORS[:k],width=.7)
        ax.set(title=f"K = {k}",xlabel="Cluster ID",xticks=list(range(k)))
        ax.tick_params(axis="x",labelsize=8)
        for bar,n in zip(bars,part.Observations):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+35,str(n),ha="center",va="bottom",fontsize=7,rotation=90)
        ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel("Trading-day observations")
    axes[0].set_ylim(0,sizes.Observations.max()*1.20)
    fig.suptitle("Cluster sizes for each candidate K",fontsize=12,fontweight="bold")
    fig.savefig(folder/"cluster_sizes_by_k.png",dpi=200);plt.close(fig)


def run(input_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir, figures = output_dir/"data", output_dir/"figures"
    data_dir.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    prices, audit = load_prices(input_path)
    features = construct_features(prices)
    eligible=common_modeling_table(features)
    standardized,assignments,scaler,models,comparison,sizes,centroids=fit_grid(eligible)
    audit.update({"selected_feature_complete_rows":int(features[FEATURES].notna().all(axis=1).sum()),
                  "modeling_complete_rows":len(eligible),
                  "rows_lost_to_rolling_warmup":int(features.Volatility_20D.isna().sum()),
                  "eligible_start":eligible.Date.min().strftime("%Y-%m-%d"),
                  "eligible_end":eligible.Date.max().strftime("%Y-%m-%d"),
                  "rows_dropped_for_unknown_next_day_target":int(features.Next_Day_Return.isna().sum()),
                  "absolute_daily_return_over_5pct_retained":int((eligible.Daily_Return.abs()>.05).sum())})
    def csv(df, name):
        df.to_csv(data_dir/name, index=False, date_format="%Y-%m-%d", float_format="%.15g")
    csv(prices, "study_prices_2015_2025.csv")
    csv(features, "clustering_candidate_features.csv")
    csv(eligible,"common_modeling_dataset.csv")
    csv(eligible[["Date"]+FEATURES], "clustering_inputs.csv")
    csv(standardized, "clustering_standardized.csv")
    csv(assignments, "cluster_assignments_k2_k6.csv")
    csv(variable_dictionary(), "variable_dictionary.csv")
    csv(comparison,"kmeans_comparison.csv")
    csv(sizes,"cluster_sizes_by_k.csv")
    scaling=pd.DataFrame({"Feature":FEATURES,"Mean":scaler.mean_,
                          "Population_SD":scaler.scale_,"Fit_observations":len(eligible)})
    csv(scaling,"scaler_parameters.csv")
    csv(centroids,"centroids_standardized_all_k.csv")
    audit["largest_absolute_return_date"]=eligible.loc[eligible.Daily_Return.abs().idxmax(),"Date"].strftime("%Y-%m-%d")
    versions={p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","matplotlib","joblib","threadpoolctl"]}
    summary={
        "owner":"Duc", "stage":"Week 2 K comparison", "authoritative_plan":"Fundamental plan.docx",
        "fit_scope":"full_period_retrospective_descriptive",
        "dataset_status":"Common prepared table supplied for group review; shared definitions and acceptance must be reconciled with teammates",
        "features":FEATURES,"k_values":K_VALUES,"settings":KMEANS_SETTINGS,
        "scaling":"StandardScaler: z=(x-mean)/population_sd, ddof=0; fit on all feature-complete 2015–2025 clustering rows",
        "feature_volatility_ddof":1,"comparison":comparison.to_dict(orient="records"),
        "all_runs_converged_before_max_iter":all(m.n_iter_<m.max_iter for m in models.values()),
        "target_class_counts":{str(int(k)):int(v) for k,v in eligible.Next_Day_Direction.value_counts().sort_index().items()},
        "predictive_split_counts":eligible.Predictive_Split.value_counts().to_dict(),
        "rows":len(eligible),"start":audit["eligible_start"],"end":audit["eligible_end"],
        "cluster_naming":"Generic IDs only; no market-regime names assigned",
        "k_status":"K=2–6 comparison complete. No final K selection or predictive claim is made.",
        "week_3_plan":"Select final K using silhouette, sizes, interpretability and stability; complete cluster profiles and peer review.",
        "predictive_reuse":"Do not reuse this full-period scaler or cluster labels for classification training/selection. A separate training-only workflow would be required.",
        "package_versions":versions,"python_version":platform.python_version(),
        "source_sha256":audit["source_sha256"],
    }
    (output_dir/"quality_log.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    (output_dir/"analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (output_dir/"model_state.json").write_text(json.dumps({"features":FEATURES,"scaler_mean":scaler.mean_.tolist(),
        "scaler_population_sd":scaler.scale_.tolist(),
        "centroids_standardized_by_k":{str(k):m.cluster_centers_.tolist() for k,m in models.items()}},indent=2),encoding="utf-8")
    create_figures(comparison,sizes,figures)
    print(json.dumps({"rows":len(eligible),"comparison":comparison.to_dict(orient="records"),"output_dir":str(output_dir)},indent=2))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"data/raw/spy_us_d.csv")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results")
    args=parser.parse_args()
    run(args.input.resolve(),args.output_dir.resolve())
