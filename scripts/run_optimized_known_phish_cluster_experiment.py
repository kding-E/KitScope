#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from web3pcapdetector.models.utils import numeric_feature_columns, select_window


PHISH_LABELS = {"phishing", "phish", "malicious", "1", "true"}
FEATURE_PREFIXES = ("base_", "role_", "phase_", "rpc_", "proto_", "flow_")
META_OR_QUALITY_PREFIXES = (
    "anchor_",
    "pcap_",
    "session_",
    "capture_",
    "window_",
    "quality_",
    "har_",
)
EXACT_EXCLUDE = {
    "base_window_s",
    "coverage_s",
    "window_s",
    "linktype",
}


@dataclass
class Candidate:
    pool: str
    topn: int
    pca: int | None
    algorithm: str
    n_clusters: int
    threshold_quantile: float


def is_phishing(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(PHISH_LABELS)


def binary_counts(y_true: Iterable[int], pred: Iterable[int]) -> dict:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(pred), dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def candidate_pool(cols: Sequence[str], pool: str) -> list[str]:
    out: list[str] = []
    for c in cols:
        if c in EXACT_EXCLUDE or c.startswith(META_OR_QUALITY_PREFIXES):
            continue
        if not c.startswith(FEATURE_PREFIXES):
            continue
        if pool == "behavior":
            out.append(c)
        elif pool == "base_role_rpc" and c.startswith(("base_", "role_", "rpc_", "proto_", "flow_")):
            out.append(c)
        elif pool == "early" and c.startswith(("phase_0_3", "phase_3_10", "role_", "rpc_", "proto_")):
            out.append(c)
        elif pool == "compact" and (
            c.startswith(("role_", "rpc_", "proto_", "flow_"))
            or "_byte_frac" in c
            or "_pkt_frac" in c
            or "_flow_frac" in c
        ):
            out.append(c)
    return out


def transform_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    x = df[list(cols)].copy()
    for c in cols:
        lc = c.lower()
        if any(tok in lc for tok in ["bytes", "pkts", "n_", "rate", "dur", "iat", "delay", "seconds", "active", "burst_sz", "flow_"]):
            vals = pd.to_numeric(x[c], errors="coerce").astype(float)
            x[c] = np.sign(vals) * np.log1p(np.abs(vals))
    return x


def select_features(
    train_phish: pd.DataFrame,
    benign: pd.DataFrame,
    cols: Sequence[str],
    topn: int,
    corr_threshold: float,
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    labelled = pd.concat([train_phish, benign], ignore_index=True)
    y = np.r_[np.ones(len(train_phish)), np.zeros(len(benign))]
    x = transform_frame(labelled, cols)
    miss = x.isna().mean()
    var = x.var(numeric_only=True)
    usable = [c for c in cols if miss[c] < 0.8 and var[c] > 1e-12]

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    z = scaler.fit_transform(imputer.fit_transform(transform_frame(labelled, usable)))
    mi = mutual_info_classif(z, y, random_state=random_state, discrete_features=False)
    ranked = pd.DataFrame({"feature": usable, "mi": mi}).sort_values("mi", ascending=False)

    ordered = ranked["feature"].tolist()
    picked: list[str] = []
    zdf = pd.DataFrame(z, columns=usable)[ordered]
    for c in ordered:
        if len(picked) >= topn:
            break
        if not picked:
            picked.append(c)
            continue
        max_corr = zdf[picked].corrwith(zdf[c]).abs().max()
        if pd.isna(max_corr) or max_corr < corr_threshold:
            picked.append(c)
    return picked, ranked


class ClusterDistanceModel:
    def __init__(self, threshold_quantile: float, variance_shrinkage: float = 0.10):
        self.threshold_quantile = float(threshold_quantile)
        self.variance_shrinkage = float(variance_shrinkage)
        self.cluster_names_: list[str] = []
        self.cluster_sizes_: dict[str, int] = {}
        self.mu_: dict[str, np.ndarray] = {}
        self.var_: dict[str, np.ndarray] = {}
        self.thresholds_: dict[str, float] = {}

    def fit(self, x: np.ndarray, labels: Sequence[int]) -> "ClusterDistanceModel":
        labels = np.asarray(labels)
        global_var = np.var(x, axis=0) + 1e-6
        for raw in sorted(np.unique(labels)):
            name = f"known_phish_c{int(raw):02d}"
            xc = x[labels == raw]
            mu = xc.mean(axis=0)
            var_emp = np.var(xc, axis=0) if len(xc) > 1 else global_var
            var = (1.0 - self.variance_shrinkage) * var_emp + self.variance_shrinkage * global_var + 1e-6
            dist = self._distance_to(xc, mu, var)
            threshold = float(np.quantile(dist, self.threshold_quantile)) if len(dist) > 1 else float(np.median(dist) if len(dist) else 0.0)
            self.cluster_names_.append(name)
            self.cluster_sizes_[name] = int(len(xc))
            self.mu_[name] = mu
            self.var_[name] = var
            self.thresholds_[name] = max(threshold, 1e-9)
        return self

    @staticmethod
    def _distance_to(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(((x - mu) ** 2) / var, axis=1))

    def predict(self, x: np.ndarray) -> pd.DataFrame:
        distances = []
        for name in self.cluster_names_:
            distances.append(self._distance_to(x, self.mu_[name], self.var_[name]))
        dmat = np.vstack(distances).T
        best_idx = dmat.argmin(axis=1)
        best = dmat[np.arange(len(x)), best_idx]
        nearest = [self.cluster_names_[i] for i in best_idx]
        threshold = np.asarray([self.thresholds_[name] for name in nearest], dtype=float)
        ratio = best / threshold
        return pd.DataFrame({
            "nearest_known_cluster": nearest,
            "nearest_distance": best,
            "cluster_threshold": threshold,
            "distance_ratio": ratio,
            "known_cluster_alert": (ratio <= 1.0).astype(int),
        })


def fit_preprocess(train: pd.DataFrame, all_parts: Sequence[pd.DataFrame], cols: Sequence[str], pca_components: int | None, random_state: int):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(transform_frame(train, cols)))
    pca = None
    if pca_components is not None:
        n = min(int(pca_components), x_train.shape[1], x_train.shape[0] - 1)
        pca = PCA(n_components=n, random_state=random_state, whiten=True)
        x_train = pca.fit_transform(x_train)

    transformed = []
    for part in all_parts:
        z = scaler.transform(imputer.transform(transform_frame(part, cols)))
        if pca is not None:
            z = pca.transform(z)
        transformed.append(z)
    return imputer, scaler, pca, transformed


def cluster_labels(x: np.ndarray, candidate: Candidate, random_state: int) -> np.ndarray:
    if candidate.algorithm == "gmm":
        return GaussianMixture(
            n_components=candidate.n_clusters,
            covariance_type="diag",
            reg_covar=1e-5,
            n_init=5,
            random_state=random_state,
        ).fit_predict(x)
    if candidate.algorithm == "kmeans":
        return KMeans(n_clusters=candidate.n_clusters, random_state=random_state, n_init=50).fit_predict(x)
    if candidate.algorithm == "agglo":
        return AgglomerativeClustering(n_clusters=candidate.n_clusters, linkage="ward").fit_predict(x)
    raise ValueError(f"unknown algorithm {candidate.algorithm}")


def score_model(model: ClusterDistanceModel, x_train: np.ndarray, x_benign: np.ndarray, x_heldout: np.ndarray) -> dict:
    pred_train = model.predict(x_train)
    pred_benign = model.predict(x_benign)
    pred_heldout = model.predict(x_heldout)
    train_ratio = pred_train["distance_ratio"].replace([np.inf, -np.inf], np.nan).fillna(np.inf).astype(float).values
    benign_ratio = pred_benign["distance_ratio"].replace([np.inf, -np.inf], np.nan).fillna(np.inf).astype(float).values
    heldout_ratio = pred_heldout["distance_ratio"].replace([np.inf, -np.inf], np.nan).fillna(np.inf).astype(float).values

    y = np.r_[np.ones(len(train_ratio)), np.zeros(len(benign_ratio))]
    score = -np.r_[train_ratio, benign_ratio]
    auroc = float(roc_auc_score(y, score))
    auprc = float(average_precision_score(y, score))
    operating_points = {}
    for fpr_target in [0.01, 0.05, 0.10, 0.20, 0.30]:
        threshold = float(np.quantile(benign_ratio, fpr_target))
        operating_points[f"benign_fpr_{fpr_target:.2f}"] = {
            "distance_ratio_threshold": threshold,
            "known_train_coverage": float((train_ratio <= threshold).mean()),
            "heldout_phish_affinity": float((heldout_ratio <= threshold).mean()),
            "benign_fpr": float((benign_ratio <= threshold).mean()),
        }
    return {
        "known_train_coverage_at_cluster_threshold": float((train_ratio <= 1.0).mean()),
        "benign_fpr_at_cluster_threshold": float((benign_ratio <= 1.0).mean()),
        "heldout_phish_affinity_at_cluster_threshold": float((heldout_ratio <= 1.0).mean()),
        "distance_auroc_known_phish_vs_benign": auroc,
        "distance_auprc_known_phish_vs_benign": auprc,
        "operating_points": operating_points,
    }


def objective(metrics: dict, n_clusters: int, target_fpr: float) -> float:
    benign_fpr = float(metrics["benign_fpr_at_cluster_threshold"])
    coverage = float(metrics["known_train_coverage_at_cluster_threshold"])
    auroc = float(metrics["distance_auroc_known_phish_vs_benign"])
    if benign_fpr > target_fpr:
        penalty = 2.5 * (benign_fpr - target_fpr)
    else:
        penalty = 0.25 * (target_fpr - benign_fpr)
    return auroc + 0.65 * coverage - 1.7 * benign_fpr - penalty + 0.015 * min(n_clusters, 16)


def attach_predictions(df: pd.DataFrame, pred: pd.DataFrame, split: str) -> pd.DataFrame:
    out = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    out["split"] = split
    out["is_phishing"] = is_phishing(out["label"]).astype(int)
    return out


def plot_result(train_pred: pd.DataFrame, benign_pred: pd.DataFrame, heldout_pred: pd.DataFrame, pca_points: dict, cluster_sizes: dict, metrics: dict, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle("Optimized Known Phishing Cluster Experiment", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    keys = list(cluster_sizes.keys())
    vals = [cluster_sizes[k] for k in keys]
    ax.bar(keys, vals, color="#4E79A7")
    ax.set_title("Known Phishing Cluster Sizes")
    ax.set_ylabel("Training phishing samples")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    train_ratio = train_pred["distance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    benign_ratio = benign_pred["distance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    heldout_ratio = heldout_pred["distance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    max_x = float(np.nanpercentile(pd.concat([train_ratio, benign_ratio, heldout_ratio]), 98))
    bins = np.linspace(0, max(1.5, max_x), 36)
    ax.hist(benign_ratio, bins=bins, alpha=0.64, color="#4E79A7", label="all benign")
    ax.hist(heldout_ratio, bins=bins, alpha=0.64, color="#F28E2B", label="heldout phishing")
    ax.hist(train_ratio, bins=bins, alpha=0.42, color="#59A14F", label="known phishing train")
    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.5, label="cluster threshold")
    ax.set_title("Distance to Nearest Known Phishing Cluster")
    ax.set_xlabel("nearest distance / cluster threshold")
    ax.set_ylabel("Samples")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    pt = pca_points
    if pt:
        ax.scatter(pt["benign"][:, 0], pt["benign"][:, 1], s=12, alpha=0.25, color="#7F7F7F", label="benign")
        ax.scatter(pt["heldout"][:, 0], pt["heldout"][:, 1], s=18, alpha=0.75, color="#F28E2B", marker="x", label="heldout phishing")
        labels = pt["cluster_labels"]
        for c in sorted(np.unique(labels)):
            mask = labels == c
            ax.scatter(pt["train"][mask, 0], pt["train"][mask, 1], s=18, alpha=0.85, label=f"c{int(c):02d}")
        ax.set_title("2D PCA View of Optimized Feature Space")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        ax.grid(alpha=0.20)
    else:
        ax.set_axis_off()

    ax = axes[1, 1]
    ops = metrics["operating_points"]
    xs, covs, helds = [], [], []
    for key, values in ops.items():
        xs.append(values["benign_fpr"])
        covs.append(values["known_train_coverage"])
        helds.append(values["heldout_phish_affinity"])
    ax.plot(xs, covs, marker="o", linewidth=2.0, color="#59A14F", label="known train coverage")
    ax.plot(xs, helds, marker="o", linewidth=2.0, color="#F28E2B", label="heldout phish affinity")
    ax.axvline(metrics["benign_fpr_at_cluster_threshold"], color="#222222", linestyle="--", linewidth=1.3, label="cluster threshold FPR")
    ax.set_title("Operating Points by Benign FPR")
    ax.set_xlabel("Benign FPR")
    ax.set_ylabel("Fraction near known clusters")
    ax.set_xlim(0, max(0.32, max(xs) + 0.02))
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features, low_memory=False)
    dyn = select_window(df, args.window).copy()
    dyn["is_phishing"] = is_phishing(dyn["label"]).astype(int)
    phishing = dyn[dyn["is_phishing"] == 1].copy()
    benign = dyn[dyn["is_phishing"] == 0].copy()
    if len(phishing) < 10 or benign.empty:
        raise ValueError("Need both phishing and benign samples")

    train_paths, heldout_paths = train_test_split(
        phishing["zip_path"].astype(str).values,
        train_size=float(args.train_phish_fraction),
        random_state=int(args.random_state),
        shuffle=True,
    )
    train_phish = phishing[phishing["zip_path"].astype(str).isin(set(train_paths))].copy()
    heldout_phish = phishing[phishing["zip_path"].astype(str).isin(set(heldout_paths))].copy()

    all_numeric = numeric_feature_columns(dyn, window_name=None, extra_exclude=["cluster", "is_phishing"])
    candidate_rows = []
    fitted_cache = {}
    for pool in args.pools:
        pool_features = candidate_pool(all_numeric, pool)
        for topn in args.topn:
            selected, ranking = select_features(train_phish, benign, pool_features, int(topn), float(args.corr_threshold), int(args.random_state))
            if len(selected) < 3:
                continue
            for pca_value in args.pca_components:
                pca_n = None if str(pca_value).lower() == "none" else int(pca_value)
                try:
                    imputer, scaler, pca, (x_train, x_benign, x_heldout) = fit_preprocess(
                        train_phish, [train_phish, benign, heldout_phish], selected, pca_n, int(args.random_state)
                    )
                except Exception:
                    continue
                for algorithm in args.algorithms:
                    for k in args.clusters:
                        if int(k) >= len(train_phish):
                            continue
                        cand_base = {"pool": pool, "topn": int(topn), "pca": pca_n, "algorithm": algorithm, "n_clusters": int(k)}
                        try:
                            labels = cluster_labels(x_train, Candidate(threshold_quantile=0.5, **cand_base), int(args.random_state))
                        except Exception:
                            continue
                        sizes = np.bincount(labels)
                        if len(sizes) != int(k) or sizes.min() < int(args.min_cluster_size):
                            continue
                        for q in args.threshold_quantiles:
                            candidate = Candidate(threshold_quantile=float(q), **cand_base)
                            model = ClusterDistanceModel(threshold_quantile=float(q)).fit(x_train, labels)
                            metrics = score_model(model, x_train, x_benign, x_heldout)
                            row = {
                                **asdict(candidate),
                                "n_features": len(selected),
                                "min_cluster_size": int(sizes.min()),
                                "max_cluster_size": int(sizes.max()),
                                **{k2: v for k2, v in metrics.items() if k2 != "operating_points"},
                            }
                            row["objective"] = objective(metrics, int(k), float(args.target_fpr))
                            candidate_rows.append(row)
                            cache_key = len(candidate_rows) - 1
                            fitted_cache[cache_key] = {
                                "candidate": candidate,
                                "selected": selected,
                                "ranking": ranking,
                                "imputer": imputer,
                                "scaler": scaler,
                                "pca": pca,
                                "x_train": x_train,
                                "x_benign": x_benign,
                                "x_heldout": x_heldout,
                                "labels": labels,
                                "model": model,
                                "metrics": metrics,
                            }

    if not candidate_rows:
        raise RuntimeError("No valid clustering candidates were produced")

    search = pd.DataFrame(candidate_rows).sort_values("objective", ascending=False).reset_index(drop=False).rename(columns={"index": "cache_key"})
    search.to_csv(out_dir / "optimized_cluster_search.csv", index=False)
    best_cache_key = int(search.iloc[0]["cache_key"])
    best = fitted_cache[best_cache_key]

    model: ClusterDistanceModel = best["model"]
    candidate: Candidate = best["candidate"]
    selected = best["selected"]
    labels = best["labels"]
    x_train = best["x_train"]
    x_benign = best["x_benign"]
    x_heldout = best["x_heldout"]
    metrics = best["metrics"]

    train_pred = attach_predictions(train_phish, model.predict(x_train), "known_phishing_train")
    benign_pred = attach_predictions(benign, model.predict(x_benign), "all_benign_test")
    heldout_pred = attach_predictions(heldout_phish, model.predict(x_heldout), "heldout_phishing_test")

    train_pred["known_cluster"] = [f"known_phish_c{int(v):02d}" for v in labels]
    train_pred.to_csv(out_dir / "known_train_predictions.csv", index=False)
    benign_pred.to_csv(out_dir / "benign_distance_predictions.csv", index=False)
    heldout_pred.to_csv(out_dir / "heldout_phish_distance_predictions.csv", index=False)
    pd.concat([train_pred, benign_pred, heldout_pred], ignore_index=True).to_csv(out_dir / "all_distance_predictions.csv", index=False)
    pd.DataFrame({"feature": selected}).to_csv(out_dir / "selected_features.csv", index=False)
    best["ranking"].to_csv(out_dir / "feature_ranking.csv", index=False)

    setup = {
        "window": args.window,
        "train_phish_fraction": float(args.train_phish_fraction),
        "random_state": int(args.random_state),
        "target_fpr_for_search": float(args.target_fpr),
        "n_train_known_phishing": int(len(train_phish)),
        "n_heldout_phishing": int(len(heldout_phish)),
        "n_all_benign_test": int(len(benign)),
        "best_candidate": asdict(candidate),
        "cluster_sizes": model.cluster_sizes_,
        "selected_feature_count": len(selected),
        "selected_features": selected,
    }
    full_metrics = {"setup": setup, **metrics}
    (out_dir / "optimized_known_phish_metrics.json").write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")

    joblib.dump({
        "candidate": candidate,
        "selected_features": selected,
        "imputer": best["imputer"],
        "scaler": best["scaler"],
        "pca": best["pca"],
        "cluster_labels": labels,
        "distance_model": model,
    }, out_dir / "optimized_known_phish_model.joblib")

    # A 2D visualization fitted only for plotting, independent of the model's optional PCA choice.
    plot_pca = PCA(n_components=2, random_state=int(args.random_state))
    plot_pca.fit(np.vstack([x_train, x_benign, x_heldout]))
    pca_points = {
        "train": plot_pca.transform(x_train),
        "benign": plot_pca.transform(x_benign),
        "heldout": plot_pca.transform(x_heldout),
        "cluster_labels": labels,
    }
    plot_result(train_pred, benign_pred, heldout_pred, pca_points, model.cluster_sizes_, metrics, out_dir / "optimized_known_phish_cluster_results.png")

    print(json.dumps(full_metrics, indent=2))
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a label-guided feature space and cluster known phishing samples away from benign traffic.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", default="dyn")
    parser.add_argument("--train-phish-fraction", type=float, default=0.70)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--corr-threshold", type=float, default=0.92)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pools", nargs="+", default=["behavior", "base_role_rpc", "early", "compact"])
    parser.add_argument("--topn", nargs="+", type=int, default=[20, 40, 80, 120])
    parser.add_argument("--pca-components", nargs="+", default=["none", "10", "20"])
    parser.add_argument("--algorithms", nargs="+", default=["gmm", "kmeans", "agglo"])
    parser.add_argument("--clusters", nargs="+", type=int, default=[3, 4, 5, 6, 8, 10, 12, 16, 20])
    parser.add_argument("--threshold-quantiles", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
