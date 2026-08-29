from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except ImportError:  # PyTorch is only required for the optional MLP backend.
    torch = None
    nn = None
else:
    # Small tabular datasets are common here; limiting threads avoids OpenMP oversubscription on shared hosts.
    torch.set_num_threads(1)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import label_to_binary, numeric_feature_columns, select_window


@dataclass
class SecondLayerConfig:
    model: str = "mlp"  # mlp | logistic | extra_trees | random_forest | hist_gradient_boosting | lightgbm | xgboost | catboost
    threshold_fpr: float = 0.05
    hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.20
    epochs: int = 200
    patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.25
    random_state: int = 42
    window_name: str = "dyn"


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for model='mlp'. Install torch or use a sklearn backend.")


class _MLP(nn.Module if nn is not None else object):
    def __init__(self, d_in: int, hidden_dims: Sequence[int], dropout: float):
        _require_torch()
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPBinaryClassifier:
    def __init__(self, config: Optional[SecondLayerConfig] = None):
        self.config = config or SecondLayerConfig()
        self.feature_columns: List[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model: Optional[_MLP] = None
        self.threshold_: float = 0.5
        self.metrics_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None, sample_weight: Optional[Sequence[float]] = None) -> "MLPBinaryClassifier":
        _require_torch()
        seed = int(self.config.random_state)
        np.random.seed(seed)
        torch.manual_seed(seed)
        df = select_window(df, self.config.window_name)
        y = label_to_binary(df["label"].values)
        if len(np.unique(y)) < 2:
            raise ValueError("Second-layer training requires both phishing and benign labels")
        if feature_columns is None:
            feature_columns = numeric_feature_columns(df, window_name=None)
        self.feature_columns = list(feature_columns)
        X = self.imputer.fit_transform(df[self.feature_columns])
        X = self.scaler.fit_transform(X).astype(np.float32)
        y = y.astype(np.float32)
        idx = np.arange(len(y))
        stratify = y if min(np.bincount(y.astype(int))) >= 2 else None
        train_idx, val_idx = train_test_split(
            idx,
            test_size=float(self.config.validation_fraction),
            random_state=int(self.config.random_state),
            stratify=stratify,
        ) if len(idx) >= 4 else (idx, idx)
        Xt = torch.tensor(X[train_idx], dtype=torch.float32)
        yt = torch.tensor(y[train_idx], dtype=torch.float32)
        Xv = torch.tensor(X[val_idx], dtype=torch.float32)
        yv = torch.tensor(y[val_idx], dtype=torch.float32)
        self.model = _MLP(X.shape[1], self.config.hidden_dims, self.config.dropout)
        pos = float((yt == 1).sum().item())
        neg = float((yt == 0).sum().item())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        best_loss = float("inf")
        best_state = None
        wait = 0
        for epoch in range(int(self.config.epochs)):
            self.model.train()
            opt.zero_grad()
            logits = self.model(Xt)
            loss = loss_fn(logits, yt)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                vloss = loss_fn(self.model(Xv), yv).item()
            if vloss < best_loss - 1e-6:
                best_loss = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= int(self.config.patience):
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        val_scores = self.predict_proba_array(X[val_idx], already_scaled=True)
        self.threshold_ = select_threshold_by_fpr(y[val_idx], val_scores, self.config.threshold_fpr)
        self.metrics_ = compute_binary_metrics(y[val_idx], val_scores, self.threshold_)
        return self

    def predict_proba_array(self, X: np.ndarray, already_scaled: bool = False) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        if not already_scaled:
            X = self.scaler.transform(self.imputer.transform(X)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy()
        logits = np.clip(logits, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        Xraw = df[self.feature_columns]
        scores = self.predict_proba_array(Xraw, already_scaled=False)
        pred = (scores >= self.threshold_).astype(int)
        return pd.DataFrame({"second_phish_score": scores, "second_pred_label": np.where(pred == 1, "phishing", "benign"), "second_threshold": self.threshold_}, index=df.index)

    def save(self, out_dir: str | pathlib.Path) -> None:
        _require_torch()
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Model not fitted")
        torch.save(self.model.state_dict(), out / "mlp_state.pt")
        joblib.dump({"imputer": self.imputer, "scaler": self.scaler}, out / "preprocess.joblib")
        meta = {
            "type": "mlp",
            "config": asdict(self.config),
            "feature_columns": self.feature_columns,
            "threshold": self.threshold_,
            "metrics": self.metrics_,
            "d_in": len(self.feature_columns),
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | pathlib.Path) -> "MLPBinaryClassifier":
        _require_torch()
        model_dir = pathlib.Path(model_dir)
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        cfg = SecondLayerConfig(**{k: tuple(v) if k == "hidden_dims" else v for k, v in meta.get("config", {}).items()})
        obj = cls(cfg)
        obj.feature_columns = meta["feature_columns"]
        obj.threshold_ = float(meta.get("threshold", 0.5))
        obj.metrics_ = meta.get("metrics", {})
        pre = joblib.load(model_dir / "preprocess.joblib")
        obj.imputer = pre["imputer"]
        obj.scaler = pre["scaler"]
        obj.model = _MLP(meta.get("d_in", len(obj.feature_columns)), cfg.hidden_dims, cfg.dropout)
        obj.model.load_state_dict(torch.load(model_dir / "mlp_state.pt", map_location="cpu"))
        obj.model.eval()
        return obj


class LogisticBinaryClassifier:
    def __init__(self, config: Optional[SecondLayerConfig] = None):
        self.config = config or SecondLayerConfig(model="logistic")
        self.feature_columns: List[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
        self.threshold_: float = 0.5
        self.metrics_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None, sample_weight: Optional[Sequence[float]] = None) -> "LogisticBinaryClassifier":
        df = select_window(df, self.config.window_name)
        y = label_to_binary(df["label"].values)
        if feature_columns is None:
            feature_columns = numeric_feature_columns(df, window_name=None)
        self.feature_columns = list(feature_columns)
        X = self.scaler.fit_transform(self.imputer.fit_transform(df[self.feature_columns]))
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)
        self.model.fit(X, y, **fit_kwargs)
        scores = self.model.predict_proba(X)[:, 1]
        self.threshold_ = select_threshold_by_fpr(y, scores, self.config.threshold_fpr)
        self.metrics_ = compute_binary_metrics(y, scores, self.threshold_)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.scaler.transform(self.imputer.transform(df[self.feature_columns]))
        scores = self.model.predict_proba(X)[:, 1]
        pred = (scores >= self.threshold_).astype(int)
        return pd.DataFrame({"second_phish_score": scores, "second_pred_label": np.where(pred == 1, "phishing", "benign"), "second_threshold": self.threshold_}, index=df.index)

    def save(self, out_dir: str | pathlib.Path) -> None:
        out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out / "logistic_classifier.joblib")
        meta = {"type": "logistic", "config": asdict(self.config), "feature_columns": self.feature_columns, "threshold": self.threshold_, "metrics": self.metrics_}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | pathlib.Path) -> "LogisticBinaryClassifier":
        return joblib.load(pathlib.Path(model_dir) / "logistic_classifier.joblib")


class GBDTEnsembleBinaryClassifier:
    """Small calibrated ensemble of modern GBDT backends.

    It averages LightGBM, XGBoost, and CatBoost probabilities when the packages
    are installed, falling back to scikit-learn histogram gradient boosting if a
    backend is unavailable.  This is intended for Layer 1 candidate windows where
    tabular features dominate and ExtraTrees was empirically weak.
    """

    def __init__(self, config: Optional[SecondLayerConfig] = None):
        self.config = config or SecondLayerConfig(model="gbdt_ensemble")
        self.feature_columns: List[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.models: list[tuple[str, object]] = []
        self.threshold_: float = 0.5
        self.metrics_: Dict[str, float] = {}

    def _member_names(self) -> list[str]:
        return ["lightgbm", "xgboost", "catboost"]

    def _build_member(self, name: str, seed_offset: int = 0):
        cfg = SecondLayerConfig(**{**asdict(self.config), "model": name, "random_state": int(self.config.random_state) + int(seed_offset)})
        return SklearnBinaryClassifier(cfg)._make_estimator()

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None, sample_weight: Optional[Sequence[float]] = None) -> "GBDTEnsembleBinaryClassifier":
        df = select_window(df, self.config.window_name)
        y = label_to_binary(df["label"].values).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("Layer 1 training requires both phishing and benign labels")
        if feature_columns is None:
            feature_columns = numeric_feature_columns(df, window_name=None)
        self.feature_columns = list(feature_columns)
        idx = np.arange(len(y))
        stratify = y if min(np.bincount(y.astype(int))) >= 2 else None
        train_idx, val_idx = train_test_split(
            idx,
            test_size=float(self.config.validation_fraction),
            random_state=int(self.config.random_state),
            stratify=stratify,
        ) if len(idx) >= 4 else (idx, idx)
        X = self.imputer.fit_transform(df[self.feature_columns])
        x_train = X[train_idx]
        x_val = X[val_idx]
        self.models = []
        for j, name in enumerate(self._member_names()):
            try:
                est = self._build_member(name, seed_offset=j)
            except ImportError:
                continue
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)[train_idx]
            elif name == "xgboost":
                yy = y[train_idx]
                pos = max(1, int((yy == 1).sum()))
                neg = max(1, int((yy == 0).sum()))
                sw = np.ones(len(yy), dtype=float)
                sw[yy == 1] = len(yy) / (2.0 * pos)
                sw[yy == 0] = len(yy) / (2.0 * neg)
                fit_kwargs["sample_weight"] = sw
            est.fit(x_train, y[train_idx], **fit_kwargs)
            self.models.append((name, est))
        if not self.models:
            fallback = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.05, random_state=int(self.config.random_state))
            fallback.fit(x_train, y[train_idx])
            self.models.append(("hist_gradient_boosting", fallback))
        val_scores = self.predict_proba_array(x_val, already_imputed=True)
        self.threshold_ = select_threshold_by_fpr(y[val_idx], val_scores, self.config.threshold_fpr)
        self.metrics_ = compute_binary_metrics(y[val_idx], val_scores, self.threshold_)
        return self

    def predict_proba_array(self, X: np.ndarray, already_imputed: bool = False) -> np.ndarray:
        if not already_imputed:
            X = self.imputer.transform(X)
        scores = []
        for _name, model in self.models:
            if hasattr(model, "predict_proba"):
                scores.append(np.asarray(model.predict_proba(X)[:, 1], dtype=float))
            else:
                raw = np.asarray(model.decision_function(X), dtype=float)
                scores.append(1.0 / (1.0 + np.exp(-np.clip(raw, -60.0, 60.0))))
        if not scores:
            raise RuntimeError("GBDT ensemble is not fitted")
        return np.clip(np.mean(np.vstack(scores), axis=0), 0.0, 1.0)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        scores = self.predict_proba_array(df[self.feature_columns], already_imputed=False)
        pred = (scores >= self.threshold_).astype(int)
        return pd.DataFrame({
            "second_phish_score": scores,
            "second_pred_label": np.where(pred == 1, "phishing", "benign"),
            "second_threshold": self.threshold_,
        }, index=df.index)

    def save(self, out_dir: str | pathlib.Path) -> None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"imputer": self.imputer, "models": self.models}, out / "gbdt_ensemble_classifier.joblib")
        meta = {
            "type": "gbdt_ensemble",
            "config": asdict(self.config),
            "feature_columns": self.feature_columns,
            "threshold": self.threshold_,
            "metrics": self.metrics_,
            "members": [name for name, _ in self.models],
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | pathlib.Path) -> "GBDTEnsembleBinaryClassifier":
        model_dir = pathlib.Path(model_dir)
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        cfg = SecondLayerConfig(**{k: tuple(v) if k == "hidden_dims" else v for k, v in meta.get("config", {}).items()})
        obj = cls(cfg)
        obj.feature_columns = meta["feature_columns"]
        obj.threshold_ = float(meta.get("threshold", 0.5))
        obj.metrics_ = meta.get("metrics", {})
        payload = joblib.load(model_dir / "gbdt_ensemble_classifier.joblib")
        obj.imputer = payload["imputer"]
        obj.models = payload["models"]
        return obj


class SklearnBinaryClassifier:
    def __init__(self, config: Optional[SecondLayerConfig] = None):
        self.config = config or SecondLayerConfig(model="extra_trees")
        self.feature_columns: List[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.model = self._make_estimator()
        self.threshold_: float = 0.5
        self.metrics_: Dict[str, float] = {}

    def _make_estimator(self):
        seed = int(self.config.random_state)
        if self.config.model == "random_forest":
            return RandomForestClassifier(
                n_estimators=600,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
        if self.config.model == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.03,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=seed,
            )
        if self.config.model == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "model='lightgbm' requires `pip install lightgbm`"
                ) from exc
            return LGBMClassifier(
                n_estimators=700,
                num_leaves=63,
                learning_rate=0.035,
                feature_fraction=0.88,
                bagging_fraction=0.90,
                bagging_freq=5,
                min_child_samples=12,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
        if self.config.model == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "model='xgboost' requires `pip install xgboost`"
                ) from exc
            return XGBClassifier(
                n_estimators=700,
                max_depth=5,
                learning_rate=0.035,
                subsample=0.90,
                colsample_bytree=0.88,
                reg_lambda=1.5,
                reg_alpha=0.05,
                min_child_weight=2.0,
                tree_method="hist",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
                use_label_encoder=False,
            )
        if self.config.model == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "model='catboost' requires `pip install catboost`"
                ) from exc
            return CatBoostClassifier(
                iterations=700,
                depth=6,
                learning_rate=0.035,
                l2_leaf_reg=3.0,
                loss_function="Logloss",
                eval_metric="Logloss",
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None, sample_weight: Optional[Sequence[float]] = None) -> "SklearnBinaryClassifier":
        df = select_window(df, self.config.window_name)
        y = label_to_binary(df["label"].values).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("Layer 1 training requires both phishing and benign labels")
        if feature_columns is None:
            feature_columns = numeric_feature_columns(df, window_name=None)
        self.feature_columns = list(feature_columns)
        idx = np.arange(len(y))
        stratify = y if min(np.bincount(y.astype(int))) >= 2 else None
        train_idx, val_idx = train_test_split(
            idx,
            test_size=float(self.config.validation_fraction),
            random_state=int(self.config.random_state),
            stratify=stratify,
        ) if len(idx) >= 4 else (idx, idx)
        x_train = self.imputer.fit_transform(df.iloc[train_idx][self.feature_columns])
        x_val = self.imputer.transform(df.iloc[val_idx][self.feature_columns])
        self.model = self._make_estimator()
        fit_kwargs = {}
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            fit_kwargs["sample_weight"] = sw[train_idx]
        elif self.config.model == "xgboost":
            yy = y[train_idx]
            pos = max(1, int((yy == 1).sum()))
            neg = max(1, int((yy == 0).sum()))
            sw = np.ones(len(yy), dtype=float)
            sw[yy == 1] = len(yy) / (2.0 * pos)
            sw[yy == 0] = len(yy) / (2.0 * neg)
            fit_kwargs["sample_weight"] = sw
        self.model.fit(x_train, y[train_idx], **fit_kwargs)
        val_scores = self.predict_proba_array(x_val, already_imputed=True)
        self.threshold_ = select_threshold_by_fpr(y[val_idx], val_scores, self.config.threshold_fpr)
        self.metrics_ = compute_binary_metrics(y[val_idx], val_scores, self.threshold_)
        return self

    def predict_proba_array(self, X: np.ndarray, already_imputed: bool = False) -> np.ndarray:
        if not already_imputed:
            X = self.imputer.transform(X)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)
        scores = np.asarray(self.model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        scores = self.predict_proba_array(df[self.feature_columns], already_imputed=False)
        pred = (scores >= self.threshold_).astype(int)
        return pd.DataFrame({
            "second_phish_score": scores,
            "second_pred_label": np.where(pred == 1, "phishing", "benign"),
            "second_threshold": self.threshold_,
        }, index=df.index)

    def save(self, out_dir: str | pathlib.Path) -> None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"imputer": self.imputer, "model": self.model}, out / "sklearn_classifier.joblib")
        meta = {
            "type": self.config.model,
            "config": asdict(self.config),
            "feature_columns": self.feature_columns,
            "threshold": self.threshold_,
            "metrics": self.metrics_,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | pathlib.Path) -> "SklearnBinaryClassifier":
        model_dir = pathlib.Path(model_dir)
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        cfg = SecondLayerConfig(**{k: tuple(v) if k == "hidden_dims" else v for k, v in meta.get("config", {}).items()})
        obj = cls(cfg)
        obj.feature_columns = meta["feature_columns"]
        obj.threshold_ = float(meta.get("threshold", 0.5))
        obj.metrics_ = meta.get("metrics", {})
        payload = joblib.load(model_dir / "sklearn_classifier.joblib")
        obj.imputer = payload["imputer"]
        obj.model = payload["model"]
        return obj


def select_threshold_by_fpr(y_true: Sequence[float], scores: Sequence[float], target_fpr: float) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    benign_scores = s[y == 0]
    if len(benign_scores) == 0:
        return 0.5
    # Highest recall under benign FPR <= target_fpr means threshold at 1-target_fpr quantile of benign scores.
    q = max(0.0, min(1.0, 1.0 - float(target_fpr)))
    return float(np.quantile(benign_scores, q))


def compute_binary_metrics(y_true: Sequence[float], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "threshold": float(threshold),
    }
    if len(np.unique(y)) == 2:
        try: out["auroc"] = float(roc_auc_score(y, s))
        except Exception: pass
        try: out["auprc"] = float(average_precision_score(y, s))
        except Exception: pass
    return out


def second_layer_from_dir(model_dir: str | pathlib.Path):
    meta = json.loads((pathlib.Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
    typ = meta.get("type", "mlp")
    if typ == "logistic":
        return LogisticBinaryClassifier.load(model_dir)
    if typ == "gbdt_ensemble":
        return GBDTEnsembleBinaryClassifier.load(model_dir)
    if typ in {"extra_trees", "random_forest", "hist_gradient_boosting", "lightgbm", "xgboost", "catboost"}:
        return SklearnBinaryClassifier.load(model_dir)
    return MLPBinaryClassifier.load(model_dir)


def train_second_layer(features_csv: str, out_dir: str, cfg: dict, model_type: str | None = None, window_name: str = "dyn"):
    df = pd.read_csv(features_csv)
    c = cfg.get("model", {}).get("second_layer", {}) if "model" in cfg else cfg
    model_type = model_type or str(c.get("model", "extra_trees"))
    conf = SecondLayerConfig(
        model=model_type,
        threshold_fpr=float(c.get("threshold_fpr", 0.05)),
        hidden_dims=tuple(c.get("hidden_dims", [128, 64])),
        dropout=float(c.get("dropout", 0.20)),
        epochs=int(c.get("epochs", 200)),
        patience=int(c.get("patience", 30)),
        learning_rate=float(c.get("learning_rate", 1e-3)),
        weight_decay=float(c.get("weight_decay", 1e-4)),
        validation_fraction=float(c.get("validation_fraction", 0.25)),
        random_state=int(c.get("random_state", 42)),
        window_name=window_name,
    )
    if model_type == "logistic":
        model = LogisticBinaryClassifier(conf)
    elif model_type == "gbdt_ensemble":
        model = GBDTEnsembleBinaryClassifier(conf)
    elif model_type in {"extra_trees", "random_forest", "hist_gradient_boosting", "lightgbm", "xgboost", "catboost"}:
        model = SklearnBinaryClassifier(conf)
    else:
        model = MLPBinaryClassifier(conf)
    model.fit(df)
    model.save(out_dir)
    return model
