"""Kit-anchored, environment-invariant MLP representation for Layer-1.

A companion score model for the deployed openset suite. It learns a
representation with three heads on the same drop-proto/stackclean feature pool
the LightGBM score model uses:

  label head   phishing vs benign (the task)
  kit head     kit-family multi-task -> pulls same-kit sessions from
               different capture days together, collapsing environment variation
               inside each kit cluster
  domain head  capture-date ERA within the training period, via a gradient
               reversal layer -> the representation is pushed to be era-invariant
               (frozen-model domain adaptation; no test data used)

Lightweight-harness screening (learn_env_invariant_features.py) showed the
score-ensemble of LightGBM + this representation beats LightGBM alone on BOTH
temporal-forward (matched-FPR recall 0.634->0.784) and random (0.937->0.946).
This module packages it as a fit/score object so run_layer1_openset_suite.py can
blend it into score_plain behind --representation-ensemble.
"""
from __future__ import annotations

import copy
import json
import pathlib
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


class _GradReverse(torch.autograd.Function if _TORCH else object):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def _make_net(d_in: int, n_kits: int, n_domains: int):
    class InvariantNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(),
            )
            self.label_head = nn.Linear(128, 1)
            self.kit_head = nn.Linear(128, max(2, n_kits))
            self.dom_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, max(2, n_domains)))

        def forward(self, x, grl_lambda: float = 0.0):
            z = self.trunk(x)
            return (self.label_head(z).squeeze(-1),
                    self.kit_head(z),
                    self.dom_head(_GradReverse.apply(z, grl_lambda)))

    return InvariantNet()


class KitDannScorer:
    """Fitted kit-anchored env-invariant scorer. .score(df) -> P(phishing) rows."""

    def __init__(self, feature_columns: list[str], mu: np.ndarray, sd: np.ndarray,
                 state_dict, n_kits: int, n_domains: int, device: str,
                 metadata: Optional[dict] = None):
        self.feature_columns = list(feature_columns)
        self.mu = mu
        self.sd = sd
        self._n_kits = n_kits
        self._n_domains = n_domains
        self._device = device
        self.metadata = dict(metadata or {})
        self._net = _make_net(len(feature_columns), n_kits, n_domains).to(device)
        self._net.load_state_dict(state_dict)
        self._net.eval()

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_columns].to_numpy(dtype="float32")
        X = np.clip((X - self.mu) / self.sd, -10, 10).astype("float32")
        out = np.empty(len(X), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.tensor(X[i:i + 8192], dtype=torch.float32).to(self._device)
                lo, _, _ = self._net(xb)
                out[i:i + 8192] = torch.sigmoid(lo).cpu().numpy()
        return out

    def embedding(self, df: pd.DataFrame) -> np.ndarray:
        """Return the frozen 128-D trunk representation for fit-only probes."""
        X = df[self.feature_columns].to_numpy(dtype="float32")
        X = np.clip((np.nan_to_num(X) - self.mu) / self.sd, -10, 10).astype("float32")
        parts = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.tensor(X[i:i + 8192], dtype=torch.float32).to(self._device)
                parts.append(self._net.trunk(xb).cpu().numpy())
        return np.concatenate(parts, axis=0) if parts else np.empty((0, 128), dtype=np.float32)

    def save(self, out_dir: str | pathlib.Path) -> None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu() for key, value in self._net.state_dict().items()}
        torch.save(state, out / "state_dict.pt")
        np.savez_compressed(out / "normalization.npz", mu=self.mu, sd=self.sd)
        payload = {
            "feature_columns": self.feature_columns,
            "n_kits": int(self._n_kits),
            "n_domains": int(self._n_domains),
            "metadata": self.metadata,
        }
        (out / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: str | pathlib.Path, device: str | None = None) -> "KitDannScorer":
        out = pathlib.Path(out_dir)
        payload = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        norm = np.load(out / "normalization.npz")
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(out / "state_dict.pt", map_location=selected_device, weights_only=True)
        return cls(
            payload["feature_columns"], norm["mu"], norm["sd"], state,
            int(payload["n_kits"]), int(payload["n_domains"]), selected_device,
            metadata=payload.get("metadata"),
        )


def _frame_matrix(df: pd.DataFrame, feature_columns: list[str], mu: np.ndarray | None = None,
                  sd: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_raw = df[feature_columns].to_numpy(dtype="float32")
    if mu is None:
        mu = np.nanmean(x_raw, axis=0)
    if sd is None:
        sd = np.nanstd(x_raw, axis=0)
        sd[sd < 1e-6] = 1.0
    x = np.clip((np.nan_to_num(x_raw) - mu) / sd, -10, 10).astype("float32")
    return x, mu, sd


def _label_loss(logits, target, *, kind: str, focal_gamma: float, focal_alpha: float,
                pos_weight: float):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if pos_weight != 1.0:
        bce = bce * torch.where(
            target > 0.5,
            torch.tensor(float(pos_weight), device=target.device),
            torch.ones_like(target),
        )
    if kind == "bce":
        return bce
    if kind == "focal":
        p = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, p, 1.0 - p)
        alpha_t = torch.where(
            target > 0.5,
            torch.tensor(float(focal_alpha), device=target.device),
            torch.tensor(float(1.0 - focal_alpha), device=target.device),
        )
        return alpha_t * torch.pow(torch.clamp(1.0 - pt, min=0.0), float(focal_gamma)) * bce
    raise ValueError(f"invalid representation label loss: {kind!r}")


def _teacher_loss(logits, target, teacher, *, kind: str, margin: float):
    if kind == "none":
        return torch.zeros_like(target)
    prob = torch.sigmoid(logits)
    if kind == "mse":
        return torch.square(prob - teacher)
    if kind == "margin":
        m = float(margin)
        pos = torch.relu(teacher - prob - m)
        neg = torch.relu(prob - teacher - m)
        return torch.where(target > 0.5, torch.square(pos), torch.square(neg))
    raise ValueError(f"invalid representation teacher loss: {kind!r}")


def _balanced_class_weights(class_ids: np.ndarray, n_classes: int, clip: float) -> np.ndarray:
    """Bounded inverse-frequency weights for an auxiliary class head."""
    ids = np.asarray(class_ids, dtype=int)
    valid = ids[(ids >= 0) & (ids < int(n_classes))]
    weights = np.ones(int(n_classes), dtype="float32")
    if len(valid) == 0 or int(n_classes) <= 0:
        return weights
    counts = np.bincount(valid, minlength=int(n_classes)).astype(float)
    nonzero = counts > 0
    raw = np.ones(int(n_classes), dtype=float)
    raw[nonzero] = len(valid) / (max(int(nonzero.sum()), 1) * counts[nonzero])
    bound = max(float(clip), 1.0)
    raw = np.clip(raw, 1.0 / bound, bound)
    raw = raw / max(float(raw[nonzero].mean()), 1e-9)
    return raw.astype("float32")


def _balanced_era_sample_weights(
    labels: np.ndarray, era_ids: np.ndarray, clip: float
) -> tuple[np.ndarray, dict]:
    """Balance label x era cells and drop eras without both label classes.

    Ordinary DANN assumes the domain is a nuisance.  In this dataset early
    public-benign captures and recent phishing captures make era a label
    shortcut.  Restricting the adversary to eras with positive and negative
    support, then equalizing their cell mass, prevents the domain head from
    winning solely through the label prior.  The task label head still sees all
    rows; only the era auxiliary loss is restricted.
    """
    labels = np.asarray(labels, dtype=int)
    eras = np.asarray(era_ids, dtype=int)
    weights = np.zeros(len(eras), dtype="float32")
    valid_eras = sorted(set(eras[eras >= 0].tolist()))
    supported = [
        era for era in valid_eras
        if set(labels[eras == era].tolist()) >= {0, 1}
    ]
    cell_counts: dict[str, int] = {}
    for era in supported:
        for label in (0, 1):
            mask = (eras == era) & (labels == label)
            cell_counts[f"{label}:{era}"] = int(mask.sum())
    positive_counts = [count for count in cell_counts.values() if count > 0]
    if positive_counts:
        total = float(sum(positive_counts))
        cells = float(len(positive_counts))
        bound = max(float(clip), 1.0)
        for era in supported:
            for label in (0, 1):
                mask = (eras == era) & (labels == label)
                count = int(mask.sum())
                if count:
                    weights[mask] = float(np.clip(total / (cells * count), 1.0 / bound, bound))
        active = weights > 0
        weights[active] /= max(float(weights[active].mean()), 1e-9)
    metadata = {
        "dated_rows": int((eras >= 0).sum()),
        "supported_rows": int((weights > 0).sum()),
        "supported_eras": [int(value) for value in supported],
        "excluded_single_label_eras": [int(value) for value in valid_eras if value not in supported],
        "label_era_cell_counts": cell_counts,
    }
    return weights, metadata


def _score_array(net, x: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float64)
    net.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i:i + batch_size], dtype=torch.float32).to(device)
            lo, _, _ = net(xb)
            out[i:i + batch_size] = torch.sigmoid(lo).cpu().numpy()
    return out


def _validation_metric(y_true: np.ndarray, prob: np.ndarray, metric: str) -> float:
    if len(np.unique(y_true)) < 2:
        eps = 1e-7
        p = np.clip(prob, eps, 1.0 - eps)
        return -float(np.mean(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))))
    if metric == "auprc":
        return float(average_precision_score(y_true, prob))
    if metric == "auroc":
        return float(roc_auc_score(y_true, prob))
    if metric == "neg_bce":
        eps = 1e-7
        p = np.clip(prob, eps, 1.0 - eps)
        return -float(np.mean(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))))
    raise ValueError(f"invalid representation early-stop metric: {metric!r}")


def load_kit_label_map(kit_labels_csv: str, canonical_path: Callable[[str], str]) -> dict:
    """cpath -> kit-family key, from a backend-kit labels CSV."""
    path = pathlib.Path(kit_labels_csv)
    if not path.exists():
        return {}
    kit = pd.read_csv(path)
    key_col = "kit_family_id" if "kit_family_id" in kit.columns else (
        "kit_family_key" if "kit_family_key" in kit.columns else (
            "static_family_key" if "static_family_key" in kit.columns else None
        )
    )
    path_col = "zip_path" if "zip_path" in kit.columns else "sample_path" if "sample_path" in kit.columns else None
    if key_col is None or path_col is None:
        return {}
    if "fit_eligible" in kit.columns:
        eligible = kit["fit_eligible"].astype(str).str.lower().isin({"1", "true", "yes"})
        kit = kit[eligible].copy()
    kit = kit[kit[key_col].fillna("").astype(str).ne("")].copy()
    kit["cpath"] = kit[path_col].map(canonical_path)
    return dict(zip(kit["cpath"], kit[key_col]))


def fit_kit_dann_scorer(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    kit_label_map: dict,
    capture_date_of: Callable[[str], str],
    canonical_path: Callable[[str], str],
    sample_weight: Optional[np.ndarray] = None,
    valid_df: Optional[pd.DataFrame] = None,
    teacher_score: Optional[np.ndarray] = None,
    valid_score_callback: Optional[Callable[[np.ndarray, int, dict], tuple[float, dict] | float]] = None,
    n_eras: int = 6,
    lam_kit: float = 0.3,
    lam_dom: float = 0.3,
    epochs: int = 25,
    batch_size: int = 2048,
    seed: int = 42,
    early_stopping: bool = False,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 1e-4,
    early_stop_metric: str = "auprc",
    early_stop_max_valid_rows: int = 100000,
    label_loss: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    positive_weight: float = 1.0,
    teacher_loss: str = "none",
    teacher_loss_weight: float = 0.0,
    teacher_margin: float = 0.0,
    balance_kit_loss: bool = False,
    balance_era_within_label: bool = False,
    auxiliary_weight_clip: float = 4.0,
) -> Optional[KitDannScorer]:
    """Train the kit-anchored env-invariant representation on train_df rows.

    Returns None if torch is unavailable. sample_weight (e.g. rank / csw weights)
    is applied to the label loss so the representation shares the score model's
    row weighting.
    """
    if not _TORCH:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = train_df.reset_index(drop=True)
    X, mu, sd = _frame_matrix(df, feature_columns)
    y = (df["label"].astype(str) == "phishing").to_numpy().astype("float32")
    if teacher_score is not None:
        teacher_score = np.asarray(teacher_score, dtype="float32")
        if len(teacher_score) != len(df):
            raise ValueError("teacher_score length must match train_df")
        teacher_score = np.clip(np.nan_to_num(teacher_score, nan=0.5), 1e-5, 1.0 - 1e-5)
    if teacher_loss != "none" and (teacher_score is None or teacher_loss_weight <= 0):
        teacher_loss = "none"
        teacher_loss_weight = 0.0

    cpath = df["zip_path"].map(canonical_path)
    kit_series = cpath.map(kit_label_map)
    kit_names = sorted(pd.unique(kit_series.dropna()))
    kit_id = {k: i for i, k in enumerate(kit_names)}
    kit = kit_series.map(kit_id).fillna(-100).astype(int).to_numpy()

    dates = pd.to_datetime(pd.Series([capture_date_of(z) for z in df["zip_path"]]),
                           format="%Y%m%d", errors="coerce")
    if int(dates.notna().sum()) >= n_eras:
        vals = dates.astype("int64").to_numpy(dtype="float64")
        valid = dates.notna().to_numpy()
        qs = np.quantile(vals[valid], np.linspace(0, 1, n_eras + 1)[1:-1])
        era = np.where(valid, np.digitize(vals, qs), -100).astype(int)
        n_domains = n_eras
    else:  # no usable dates -> disable domain head
        era = np.full(len(df), -100, dtype=int)
        n_domains = 2
        lam_dom = 0.0

    w = (np.ones(len(df), dtype="float32") if sample_weight is None
         else np.asarray(sample_weight, dtype="float32"))
    w = w * (len(w) / max(w.sum(), 1e-9))

    net = _make_net(len(feature_columns), len(kit_names), n_domains).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    kit_class_weights = _balanced_class_weights(
        kit, len(kit_names), float(auxiliary_weight_clip)
    ) if balance_kit_loss else np.ones(max(len(kit_names), 1), dtype="float32")
    kit_ce = nn.CrossEntropyLoss(
        ignore_index=-100,
        weight=(torch.tensor(kit_class_weights, dtype=torch.float32).to(device)
                if len(kit_names) >= 2 else None),
    )
    if balance_era_within_label:
        era_weights, era_balance_metadata = _balanced_era_sample_weights(
            y.astype(int), era, float(auxiliary_weight_clip)
        )
    else:
        era_weights = (era >= 0).astype("float32")
        era_balance_metadata = {
            "dated_rows": int((era >= 0).sum()),
            "supported_rows": int((era >= 0).sum()),
            "supported_eras": sorted(set(era[era >= 0].astype(int).tolist())),
            "excluded_single_label_eras": [],
            "label_era_cell_counts": {},
        }
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    wt = torch.tensor(w, dtype=torch.float32)
    kt = torch.tensor(kit, dtype=torch.long)
    dt = torch.tensor(era, dtype=torch.long)
    dwt = torch.tensor(era_weights, dtype=torch.float32)
    tt = torch.tensor(teacher_score if teacher_score is not None else np.full(len(df), 0.5, dtype="float32"),
                      dtype=torch.float32)
    n = len(Xt)
    has_kit = len(kit_names) >= 2 and (kit >= 0).any()
    valid_pack = None
    if early_stopping and valid_df is not None and len(valid_df) > 0:
        vdf = valid_df.reset_index(drop=True)
        if (
            valid_score_callback is None
            and early_stop_max_valid_rows > 0
            and len(vdf) > early_stop_max_valid_rows
        ):
            rng = np.random.default_rng(seed + 29)
            vy = (vdf["label"].astype(str) == "phishing").to_numpy()
            pos_idx = np.where(vy)[0]
            neg_idx = np.where(~vy)[0]
            max_pos = min(len(pos_idx), early_stop_max_valid_rows // 2)
            max_neg = max(0, early_stop_max_valid_rows - max_pos)
            keep = np.concatenate([
                rng.choice(pos_idx, size=max_pos, replace=False) if len(pos_idx) > max_pos else pos_idx,
                rng.choice(neg_idx, size=min(len(neg_idx), max_neg), replace=False) if len(neg_idx) > max_neg else neg_idx,
            ])
            keep = np.sort(keep)
            vdf = vdf.iloc[keep].reset_index(drop=True)
        Xv, _, _ = _frame_matrix(vdf, feature_columns, mu, sd)
        yv = (vdf["label"].astype(str) == "phishing").to_numpy().astype("float32")
        valid_pack = (Xv, yv)

    best_state = None
    best_epoch = -1
    best_score = -np.inf
    stale_epochs = 0
    history: list[dict] = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        lam = lam_dom * min(1.0, (ep + 1) / max(1, epochs // 2))
        train_loss_sum = 0.0
        train_seen = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx].to(device)
            # The scalar belongs on the era loss once.  The GRL reverses the
            # encoder gradient with unit magnitude; multiplying the CE term by
            # ``lam`` implements the stated -lambda_e L_e saddle objective.
            lo, ko, do = net(xb, grl_lambda=(1.0 if lam > 0 else 0.0))
            wb = wt[idx].to(device)
            yb = yt[idx].to(device)
            loss_vec = _label_loss(
                lo, yb,
                kind=label_loss,
                focal_gamma=float(focal_gamma),
                focal_alpha=float(focal_alpha),
                pos_weight=float(positive_weight),
            )
            if teacher_loss != "none" and teacher_loss_weight > 0:
                loss_vec = loss_vec + float(teacher_loss_weight) * _teacher_loss(
                    lo, yb, tt[idx].to(device), kind=teacher_loss, margin=float(teacher_margin)
                )
            loss = (loss_vec * wb).mean()
            if lam_kit > 0 and has_kit:
                loss = loss + lam_kit * kit_ce(ko, kt[idx].to(device))
            if lam > 0:
                domain_target = dt[idx].to(device)
                domain_weight = dwt[idx].to(device)
                domain_loss = nn.functional.cross_entropy(
                    do, domain_target, ignore_index=-100, reduction="none"
                )
                active_domain = (domain_target >= 0) & (domain_weight > 0)
                if bool(active_domain.any()):
                    domain_loss = (
                        domain_loss[active_domain] * domain_weight[active_domain]
                    ).sum() / domain_weight[active_domain].sum().clamp_min(1e-9)
                    loss = loss + lam * domain_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.detach().cpu()) * len(idx)
            train_seen += len(idx)
        row = {"epoch": int(ep + 1), "train_loss": float(train_loss_sum / max(train_seen, 1))}
        if valid_pack is not None:
            prob = _score_array(net, valid_pack[0], device)
            callback_detail = None
            if valid_score_callback is not None:
                callback_out = valid_score_callback(prob, ep + 1, row)
                if isinstance(callback_out, tuple):
                    val_score = float(callback_out[0])
                    callback_detail = callback_out[1]
                else:
                    val_score = float(callback_out)
            else:
                val_score = _validation_metric(valid_pack[1], prob, early_stop_metric)
            row[f"valid_{early_stop_metric}"] = float(val_score)
            if callback_detail is not None:
                row["valid_selection"] = callback_detail
            if val_score > best_score + float(early_stop_min_delta):
                best_score = val_score
                best_epoch = ep + 1
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append(row)
            if stale_epochs >= int(early_stop_patience):
                break
        else:
            history.append(row)

    net.eval()
    if best_state is not None:
        net.load_state_dict(best_state)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    metadata = {
        "epochs_requested": int(epochs),
        "epochs_ran": int(history[-1]["epoch"] if history else 0),
        "early_stopping": bool(valid_pack is not None),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_metric": str(early_stop_metric),
        "best_epoch": int(best_epoch if best_epoch > 0 else (history[-1]["epoch"] if history else 0)),
        "best_valid_score": (float(best_score) if np.isfinite(best_score) else None),
        "label_loss": str(label_loss),
        "focal_gamma": float(focal_gamma),
        "focal_alpha": float(focal_alpha),
        "positive_weight": float(positive_weight),
        "teacher_loss": str(teacher_loss),
        "teacher_loss_weight": float(teacher_loss_weight),
        "teacher_margin": float(teacher_margin),
        "balance_kit_loss": bool(balance_kit_loss),
        "balance_era_within_label": bool(balance_era_within_label),
        "auxiliary_weight_clip": float(auxiliary_weight_clip),
        "kit_class_weight_min": float(kit_class_weights.min()) if len(kit_class_weights) else None,
        "kit_class_weight_max": float(kit_class_weights.max()) if len(kit_class_weights) else None,
        "era_balance": era_balance_metadata,
        "n_train_rows": int(len(df)),
        "n_valid_rows": int(0 if valid_pack is None else len(valid_pack[1])),
        "n_kit_labels": int(len(kit_names)),
        "n_domains": int(n_domains),
        "history": history,
    }
    return KitDannScorer(feature_columns, mu, sd, state, len(kit_names), n_domains, device, metadata=metadata)
