from argparse import Namespace
import logging
import os
import random
import time
import warnings
from typing import Any, Callable, Dict, Optional
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from model.utils import tune_hyper_parameters, is_constant_prediction
from tabpfn import TabPFNClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from skrub import TableVectorizer
from model.utils import get_method
from model.lib.data import (
    data_enc_process,
    data_label_process,
    data_nan_process,
    get_dataset,
)
from tabicl import TabICLClassifier

from scipy.sparse import csr_matrix
from sklearn.utils import Bunch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from cache_io import (
    append_manifest,
    cache_exists,
    cache_path,
    conformal_from_arrays,
    conformal_from_cache,
    package_versions,
    read_cache,
    write_cache,
)

import wandb
from mapie.utils import train_conformalize_test_split
from mapie.classification import SplitConformalClassifier, CrossConformalClassifier
from mapie.metrics.classification import (
    classification_coverage_score,
    classification_mean_width_score,
    classification_ssc_score,
)
from mapie.metrics.calibration import expected_calibration_error, top_label_ece

try:
    from mapie.estimator.classifier import EnsembleClassifier
except Exception:  # noqa: BLE001
    EnsembleClassifier = None
else:
    if EnsembleClassifier and not hasattr(EnsembleClassifier, "__sklearn_tags__"):

        def _mapie_sklearn_tags(self):
            return BaseEstimator.__sklearn_tags__(self)

        EnsembleClassifier.__sklearn_tags__ = _mapie_sklearn_tags  # type: ignore[assignment]


# pandas 3 (hplr-new / venv-new: pandas 3.0.5) removed SettingWithCopyWarning.
try:
    from pandas.errors import SettingWithCopyWarning
except Exception:  # noqa: BLE001
    try:
        from pandas.core.common import SettingWithCopyWarning
    except Exception:  # noqa: BLE001
        SettingWithCopyWarning = None

if SettingWithCopyWarning is not None:
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from tabm import TabM

DATA_DIR = "/content/MyDrive/MyDrive/Datasets/Rain_in_Australia"


def set_seed(seed: int = 42):
    """
    Sets the random seed for various libraries to ensure reproducibility.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch on CPU and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # # Configure PyTorch to use deterministic algorithms
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Set the PYTHONHASHSEED environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to {seed}")


class TalentScikitWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper to make TALENT models compatible with the scikit-learn API.
    """

    def __init__(self, talent_method, info, n_cat_features=0):
        self.talent_method = talent_method
        self.info = info
        self.n_cat_features = n_cat_features
        self._is_fitted = False
        self.config = talent_method.args.config
        # This attribute is necessary for scikit-learn to recognize this as a classifier.
        self._estimator_type = "classifier"

    def _split_features(self, X):
        """Splits the input X into numerical and categorical parts."""
        if self.n_cat_features == 0:
            N = X
            C = None
        elif self.n_cat_features == X.shape[1]:
            N = None
            C = X
        else:
            C = X[:, : self.n_cat_features]
            N = X[:, self.n_cat_features :]
        return N, C

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fits the underlying TALENT model.

        When ``X_val``/``y_val`` are given they become TALENT's ``val`` split,
        which epoch-trained methods use for best-epoch selection and early
        stopping. Without them ``val`` aliases ``train`` (TALENT then selects
        the epoch of highest *training* accuracy - only acceptable for methods
        that do not early-stop, e.g. GBDTs, kNN, in-context TFMs).
        """
        N, C = self._split_features(X)
        if X_val is None:
            N_val, C_val, y_val = N, C, y
        else:
            N_val, C_val = self._split_features(X_val)
            assert X_val is not X, "validation split must be held out"

        N_dict = {"train": N, "val": N_val} if N is not None else None
        C_dict = {"train": C, "val": C_val} if C is not None else None
        y_dict = {"train": y, "val": y_val}

        fit_data = (N_dict, C_dict, y_dict)

        self.talent_method.fit(fit_data, self.info, train=True, config=self.config)
        self._is_fitted = True
        self.classes_ = self.talent_method.label_encoder.classes_
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for X.
        """
        if not self._is_fitted:
            raise RuntimeError("This model is not fitted yet.")

        N, C = self._split_features(X)
        N_dict = {"test": N} if N is not None else None
        C_dict = {"test": C} if C is not None else None
        # The `y` part is not used for prediction but required by the API.
        y_dict = {"test": np.zeros((X.shape[0],), dtype=int)}

        logits = self.talent_method.predict(
            (N_dict, C_dict, y_dict), self.info, model_name="best-val"
        )

        if isinstance(logits, torch.Tensor):
            logits = np.array(logits.cpu())

        if np.any((logits < 0) | (logits > 1)) or (
            not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)
        ):
            exps = np.exp(
                logits - np.max(logits, axis=1, keepdims=True)
            )  # stabilize by subtracting max
            logits = exps / np.sum(exps, axis=1, keepdims=True)

        return logits

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def _package_version_tuple(package: str):
    import importlib.metadata as md
    import re

    v = md.version(package)
    return v, tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _hf_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")


class PipTFMClassifier(BaseEstimator, ClassifierMixin):
    """A pip tabular foundation model behind the ``TalentScikitWrapper`` API.

    ``fit(X, y, X_val=None, y_val=None)`` / ``predict_proba(X)`` with ``X``
    laid out as ``[C | N]`` (the ``n_cat_features`` categorical columns first,
    exactly what ``evaluate_talent_dataset`` builds). The preprocessing is the
    vendored in-context path of ``model/methods/PFN_v2.py``: TALENT's
    ``data_nan_process`` (mean imputation, ``new`` category for missing
    strings), ``data_label_process`` (LabelEncoder), ``data_enc_process``
    with ``cat_policy="indices"`` (ordinal codes), then the model sees
    ``[N | C]`` with the categorical block flagged through
    ``categorical_features_indices`` (TabPFN) - TabICL takes plain arrays,
    like the vendored v1 wrapper.

    Exposes ``label_encoder``, ``classes_``, ``model`` (the fitted estimator),
    ``trlog`` (empty) and ``model_info`` (version / checkpoint / ensemble
    settings) so the caller treats it like a TALENT method.
    """

    def __init__(
        self,
        model_name,
        device,
        seed,
        n_cat_features=0,
        n_estimators=None,
        softmax_temperature=None,
        num_nan_policy="mean",
        cat_nan_policy="new",
    ):
        if model_name not in NEW_TFM_SPECS:
            raise ValueError(f"unknown pip TFM {model_name!r}; known: {sorted(NEW_TFM_SPECS)}")
        self.model_name = model_name
        self.spec = NEW_TFM_SPECS[model_name]
        self.device = device
        self.seed = int(seed)
        self.n_cat_features = int(n_cat_features)
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.num_nan_policy = num_nan_policy
        self.cat_nan_policy = cat_nan_policy
        self._is_fitted = False
        self._estimator_type = "classifier"
        self.trlog = {}
        self.model = None
        self.model_info = {}

    # -- data layout -------------------------------------------------------
    def _split_features(self, X):
        if self.n_cat_features == 0:
            return X, None
        if self.n_cat_features == X.shape[1]:
            return None, X
        return X[:, self.n_cat_features :], X[:, : self.n_cat_features]

    @staticmethod
    def _assemble(N, C):
        """[N | C] plus the categorical indices, as model/methods/PFN_v2.py."""
        if N is not None and C is not None:
            X = np.concatenate((N, C), axis=1)
            cat_indices = list(range(N.shape[1], N.shape[1] + C.shape[1]))
        elif N is None and C is not None:
            X, cat_indices = C, list(range(C.shape[1]))
        else:
            X, cat_indices = N, []
        return X, cat_indices

    # -- checkpoint / constructor ------------------------------------------
    def effective_n_estimators(self) -> int:
        return int(self.spec["n_estimators"] if self.n_estimators is None else self.n_estimators)

    def effective_softmax_temperature(self) -> float:
        return float(
            self.spec["softmax_temperature"]
            if self.softmax_temperature is None
            else self.softmax_temperature
        )

    def _check_package(self):
        version, tup = _package_version_tuple(self.spec["package"])
        if tup < self.spec["min_version"]:
            raise RuntimeError(
                f"{self.model_name} needs {self.spec['package']}>="
                f"{'.'.join(map(str, self.spec['min_version']))}, found {version} "
                f"(run it from the hplr-new / venv-new environment)"
            )
        return version

    def resolve_checkpoint(self) -> Optional[str]:
        """Path of the checkpoint on disk, or a loud FileNotFoundError.

        TabPFN: ``$TABPFN_MODEL_CACHE_DIR/<default filename>`` (the package's
        own ``prepend_cache_path``); the file is loaded with ``torch.load`` and
        neither the network nor ``auth_token`` is touched when it exists.
        TabICL: the HF hub cache (``hf_hub_download(local_files_only=True)``);
        returns None when it is absent and downloading is allowed.
        """
        ckpt = self.spec["checkpoint"]
        if self.spec["package"] == "tabfm":
            from huggingface_hub import hf_hub_download

            try:
                return hf_hub_download(self.spec["hf_repo"], ckpt, local_files_only=True)
            except Exception:
                return None
        if self.spec["package"] == "exaonetabular":
            from huggingface_hub import hf_hub_download

            try:
                return hf_hub_download(self.spec["hf_repo"], ckpt, local_files_only=True)
            except Exception:
                return None
        if self.spec["package"] == "Xiaomi-TabLDM":
            from huggingface_hub import hf_hub_download

            try:
                return hf_hub_download(self.spec["hf_repo"], ckpt, local_files_only=True)
            except Exception:
                return None
        if self.spec["package"] == "tabpfn":
            from tabpfn.model_loading import ModelSource, prepend_cache_path

            src = {
                "v3": ModelSource.get_classifier_v3,
                "v2.5": ModelSource.get_classifier_v2_5,
                "v2.6": ModelSource.get_classifier_v2_6,
            }[self.spec["model_version"]]()
            # A spec may deliberately pin a sibling checkpoint of the same
            # version (a within-family control). It must still come from the
            # version's own repo; only the filename is allowed to differ.
            if self.spec.get("checkpoint_is_variant"):
                assert src.repo_id == self.spec["hf_repo"], (
                    src.repo_id, self.spec["hf_repo"]
                )
            else:
                assert src.default_filename == ckpt, (src.default_filename, ckpt)
            path = prepend_cache_path(ckpt)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{self.model_name}: TabPFN {self.spec['model_version']} checkpoint "
                    f"'{ckpt}' not found at {path}. Copy the file there (or point "
                    f"TABPFN_MODEL_CACHE_DIR at a directory holding it); the gated "
                    f"download from HF {src.repo_id} needs a Prior Labs login on a "
                    f"machine with internet - this job will not attempt it."
                )
            return path
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(self.spec["hf_repo"], ckpt, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            if _hf_offline():
                raise FileNotFoundError(
                    f"{self.model_name}: TabICL checkpoint '{ckpt}' is not in the HF hub "
                    f"cache ({os.environ.get('HF_HOME', '~/.cache/huggingface')}/hub/"
                    f"models--{self.spec['hf_repo'].replace('/', '--')}) and "
                    f"HF_HUB_OFFLINE is set - copy the snapshot there."
                ) from exc
            return None

    def _device_str(self) -> str:
        return str(self.device)

    def _build(self, cat_indices):
        package_version = self._check_package()
        path = self.resolve_checkpoint()
        n_est = self.effective_n_estimators()
        # EXAONE has no temperature knob; its spec declares None and
        # effective_softmax_temperature() would raise on float(None).
        temp = (
            None
            if self.spec["softmax_temperature"] is None
            and self.softmax_temperature is None
            else self.effective_softmax_temperature()
        )
        if self.spec["package"] == "tabfm":
            import torch as _torch

            from tabfm import TabFMClassifier as _TabFM
            from tabfm import tabfm_v1_0_0_pytorch as _tabfm_backend

            # The backend defaults to bfloat16, which does not exist before
            # Ampere. Choosing by compute capability would work but would make
            # a cell depend on which card it landed on, and this campaign
            # spans V100 and A100 nodes. The dtype is pinned in the spec
            # instead, so every cell is the same computation wherever it runs.
            _dtype = getattr(_torch, self.spec["compute_dtype"])
            _base = _tabfm_backend.load(
                model_type="classification",
                device=self._device_str(),
                dtype=_dtype,
            )
            self.model = _TabFM(
                model=_base,
                n_estimators=n_est,
                softmax_temperature=temp,
                average_logits=True,
                random_state=self.seed,
                verbose=False,
            )
            averages_logits = True
            average_before_softmax = None
        elif self.spec["package"] == "exaonetabular":
            from exaonetabular import EXAONETabularClassifier as _EXA

            kw = dict(device=self._device_str(), seed=self.seed)
            if n_est is not None:
                kw["ensemble_count"] = int(n_est)
            if path is not None:
                # A cached checkpoint keeps the compute node off the network;
                # from_pretrained() would otherwise reach for the Hub.
                kw["weights"] = path
            self.model = _EXA.from_pretrained(**kw)
            averages_logits = None
            average_before_softmax = None
        elif self.spec["package"] == "Xiaomi-TabLDM":
            import inspect

            from tabldm import TabLDMClassifier as _TabLDM

            kw = dict(
                device=self._device_str(),
                random_state=self.seed,
                categorical_indices=cat_indices or None,
                verbose=False,
            )
            _pp = inspect.signature(_TabLDM.__init__).parameters
            if "n_estimators" in _pp:
                kw["n_estimators"] = n_est
            if "softmax_temperature" in _pp:
                kw["softmax_temperature"] = temp
            if path is not None and "checkpoint_version" in _pp:
                kw["checkpoint_version"] = self.spec["checkpoint"]
            self.model = _TabLDM(**kw)
            averages_logits = None
            average_before_softmax = None
        elif self.spec["package"] == "tabicl":
            import inspect

            from tabicl import TabICLClassifier as _TabICL

            if "checkpoint_version" not in inspect.signature(_TabICL).parameters:
                raise RuntimeError("tabiclv2 needs the tabicl>=2.1 TabICLClassifier")
            self.model = _TabICL(
                n_estimators=n_est,
                softmax_temperature=temp,
                average_logits=True,
                checkpoint_version=self.spec["checkpoint"],
                model_path=None,
                allow_auto_download=path is None,
                device=self._device_str(),
                random_state=self.seed,
                verbose=False,
            )
            averages_logits = True
            average_before_softmax = None
        else:
            from tabpfn import TabPFNClassifier as _TabPFN

            self.model = _TabPFN(
                n_estimators=n_est,
                # keep n_estimators exactly as requested (8.x would otherwise
                # raise it on wide datasets); the effective value is recorded
                auto_scale_n_estimators=False,
                categorical_features_indices=cat_indices or None,
                softmax_temperature=temp,
                average_before_softmax=False,
                model_path=path,
                device=self._device_str(),
                ignore_pretraining_limits=True,
                random_state=self.seed,
            )
            averages_logits = False
            average_before_softmax = False
        self.model_info = dict(
            model_version=self.spec["model_version"],
            checkpoint=self.spec["checkpoint"],
            checkpoint_path=path,
            package=self.spec["package"],
            package_version=package_version,
            n_estimators=n_est,
            n_estimators_effective=n_est,
            softmax_temperature=temp,
            average_logits=averages_logits,
            average_before_softmax=average_before_softmax,
            auto_scale_n_estimators=False if self.spec["package"] == "tabpfn" else None,
            categorical_features_indices=list(cat_indices),
        )

    # -- sklearn API --------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        N, C = self._split_features(X)
        N_d = {"train": N, "val": N} if N is not None else None
        C_d = {"train": C, "val": C} if C is not None else None
        y_d = {"train": y, "val": y}
        N_d, C_d, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(
            N_d, C_d, self.num_nan_policy, self.cat_nan_policy
        )
        y_d, self.y_info, self.label_encoder = data_label_process(y_d, False)
        N_d, C_d, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(
            N_d, C_d, "indices"
        )
        X_fit, cat_indices = self._assemble(
            N_d["train"] if N_d is not None else None,
            C_d["train"] if C_d is not None else None,
        )
        self._build(cat_indices)
        self.model.fit(X_fit, y_d["train"])
        self.classes_ = self.label_encoder.classes_
        eff = getattr(self.model, "n_estimators_", None)
        if eff is not None:
            self.model_info["n_estimators_effective"] = int(eff)
        mp = getattr(self.model, "model_path_", None)
        if mp is not None:
            self.model_info["checkpoint_path"] = str(mp)
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("This model is not fitted yet.")
        N, C = self._split_features(X)
        N_d = {"test": N} if N is not None else None
        C_d = {"test": C} if C is not None else None
        N_d, C_d, _, _, _ = data_nan_process(
            N_d,
            C_d,
            self.num_nan_policy,
            self.cat_nan_policy,
            self.num_new_value,
            self.imputer,
            self.cat_new_value,
        )
        N_d, C_d, _, _, _ = data_enc_process(
            N_d, C_d, "indices", None, self.ord_encoder, self.mode_values, self.cat_encoder
        )
        X_t, _ = self._assemble(
            N_d["test"] if N_d is not None else None,
            C_d["test"] if C_d is not None else None,
        )
        p = np.asarray(self.model.predict_proba(X_t), dtype=np.float64)
        if np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(axis=-1), 1, atol=1e-5):
            exps = np.exp(p - np.max(p, axis=1, keepdims=True))
            p = exps / np.sum(exps, axis=1, keepdims=True)
        return p

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def ensemble_info(model_name, talent_method) -> Dict[str, Any]:
    """Ensemble size / temperature / averaging / checkpoint actually used by a
    fitted model, for the npz ``meta``. Pip TFMs report ``model_info``;
    vendored TALENT methods are read through their ``.model`` attributes with
    the constructor literals of model/methods/*.py as the fallback."""
    if isinstance(talent_method, PipTFMClassifier):
        return dict(talent_method.model_info)

    inner = getattr(talent_method, "model", None)
    defaults = VENDORED_TFM_DEFAULTS.get(model_name, {})
    info: Dict[str, Any] = dict(
        model_version=defaults.get("model_version"),
        checkpoint=None,
        checkpoint_path=None,
        package="vendored" if model_name in IN_CONTEXT_MODELS else None,
        package_version=None,
        n_estimators=defaults.get("n_estimators"),
        n_estimators_effective=None,
        softmax_temperature=defaults.get("softmax_temperature"),
        average_logits=None,
        average_before_softmax=None,
        auto_scale_n_estimators=None,
        categorical_features_indices=None,
    )
    if inner is None:
        return info
    for attr in ("n_estimators", "N_ensemble_configurations"):
        if hasattr(inner, attr):
            info["n_estimators"] = getattr(inner, attr)
            break
    info["n_estimators_effective"] = getattr(inner, "n_estimators_", info["n_estimators"])
    if hasattr(inner, "softmax_temperature"):
        info["softmax_temperature"] = inner.softmax_temperature
    elif hasattr(inner, "temperature"):  # tabpfn v1: loaded from the checkpoint
        t = inner.temperature
        info["softmax_temperature"] = float(t) if t is not None else None
    if hasattr(inner, "average_before_softmax"):  # TabPFN family
        info["average_before_softmax"] = bool(inner.average_before_softmax)
        info["average_logits"] = bool(inner.average_before_softmax)
    if hasattr(inner, "average_logits"):  # TabICL family
        info["average_logits"] = bool(inner.average_logits)
    if hasattr(inner, "auto_scale_n_estimators"):
        info["auto_scale_n_estimators"] = bool(inner.auto_scale_n_estimators)
    if hasattr(inner, "categorical_features_indices"):
        cfi = inner.categorical_features_indices
        info["categorical_features_indices"] = None if cfi is None else list(cfi)
    for attr in ("model_path_", "model_path"):
        mp = getattr(inner, attr, None)
        if mp is not None and str(mp) != "auto":
            info["checkpoint_path"] = str(mp)
            info["checkpoint"] = os.path.basename(str(mp))
            break
    if info["checkpoint"] is None and getattr(inner, "checkpoint_version", None):
        info["checkpoint"] = str(inner.checkpoint_version)
    if model_name == "tabpfn" and info["checkpoint"] is None:
        info["checkpoint"] = "prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
    return info


def load(name) -> Optional[np.ndarray]:
    p = os.path.join(DATA_DIR, name)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


# ---- build X by concatenating [C | N] ----
def concat_features(C_part, N_part):
    parts = [p for p in (C_part, N_part) if p is not None]
    if not parts:
        raise ValueError("No features found (need at least C_* or N_*).")
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def evaluate_classification(y_pred, y_test, y_pred_set, y_prob):
    """Evaluate classification models (binary & multiclass)."""
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    if y_prob.shape[1] == 2:  # Binary
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:  # Multiclass
        auc = roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted")

    cr = classification_coverage_score(y_test, y_pred_set)
    mwc = classification_mean_width_score(y_pred_set)
    sscs = classification_ssc_score(y_test, y_pred_set)
    # ``ece`` keeps the ESANN-era call for continuity. For binary tasks it
    # hands MAPIE the full (n, 2) matrix, which bins the top-label confidence
    # against the class-1 frequency rather than against correctness.
    # ``ece_top_label_15`` is the top-label ECE with 15 bins for both branches.
    if y_prob.shape[1] == 2:
        ece = expected_calibration_error(y_test, y_prob)
    else:
        ece = top_label_ece(y_test, y_prob)
    ece_fixed = top_label_ece(y_test, y_prob, num_bins=15)
    labels = np.arange(y_prob.shape[1])
    onehot = (np.asarray(y_test)[:, None] == labels[None, :]).astype(float)
    brier = float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))
    nll = float(log_loss(y_test, y_prob, labels=labels))
    set_sizes = np.asarray(y_pred_set)[:, :, 0].sum(axis=1)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
        "coverage_rate": cr[0].item(),
        "mean_width": mwc[0].item(),
        "ssc_score": sscs[0].item(),
        "ece": ece,
        "ece_top_label_15": ece_fixed,
        "brier": brier,
        "nll": nll,
        "empty_set_rate": float(np.mean(set_sizes == 0)),
        "max_prob_mean": float(np.mean(y_prob.max(axis=1))),
    }


def clean_col(col):
    return (
        col.replace("<=", "_le_")
        .replace("<", "_lt_")
        .replace("[", "_")
        .replace("]", "_")
    )


def tune_model_with_optuna(
    model_class,
    trial_params: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=25,
):
    """
    Tunes a model's hyperparameters using Optuna.

    Args:
        model_class: The class of the model to tune (e.g., LGBMClassifier).
        trial_params: A dictionary defining the search space for Optuna.
        X_train, y_train: Training data.
        X_val, y_val: Validation data for evaluation.
        n_trials: The number of optimization trials to run.

    Returns:
        A dictionary containing the best hyperparameters found.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            name: getattr(trial, f"suggest_{sugg_type}")(name, **kwargs)
            for name, (sugg_type, kwargs) in trial_params.items()
        }
        model = model_class(**params)

        # Suppress verbose output from models during tuning
        if "verbose" in model.get_params():
            model.set_params(verbose=-1)
        if "verbosity" in model.get_params():
            model.set_params(verbosity=0)

        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        score = accuracy_score(y_val, y_pred_val)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


DEEP_MODELS = [
    "mlp",
    "resnet",
    "ftt",
    "node",
    "autoint",
    "tabpfn",
    "tangos",
    "saint",
    "tabcaps",
    "tabnet",
    "snn",
    "ptarl",
    "danets",
    "dcn2",
    "tabtransformer",
    "dnnr",
    "switchtab",
    "grownet",
    "tabr",
    "modernNCA",
    "hyperfast",
    "bishop",
    "realmlp",
    "protogate",
    "mlp_plr",
    "excelformer",
    "grande",
    "amformer",
    "tabptm",
    "trompt",
    "tabm",
    "PFN-v2",
    "t2gformer",
    "tabautopnpnet",
    "tabicl",
]

CLASSICAL_MODELS = [
    "LogReg",
    #    "NCM",
    "RandomForest",
    "xgboost",
    "catboost",
    "lightgbm",
    #    "svm",
    "knn",
    "NaiveBayes",
    "dummy",
    "LinearRegression",
]


def get_talent_args(
    dataset_name, dataset_path, model_type, seed, n_trials, work_dir="./",
    hpo_objective="accuracy",
):
    """Creates a mock args object for TALENT functions.

    ``work_dir`` is TALENT's ``save_path``: every ``best-val-{seed}.pth``,
    ``epoch-last-{seed}.pth``, ``trlog`` and ``{model}-tuned.json`` lands
    there, so two concurrent jobs must never share it.

    ``hpo_objective`` (``accuracy`` | ``logloss``) is read by
    ``model.utils.tune_hyper_parameters`` for classification tasks.
    """
    args = Bunch()
    args.dataset = dataset_name
    args.dataset_path = dataset_path
    args.model_type = model_type
    args.seed = seed
    os.makedirs(work_dir, exist_ok=True)
    args.save_path = work_dir
    args.use_float = True
    # Add other default args that TALENT might need
    args.n_trials = n_trials
    args.hpo_objective = hpo_objective

    import importlib.resources as pkg_resources
    import json

    default_para_model = {}
    opt_space_model = {}
    model_config = {}

    config_found = False
    for config_type in ["deep", "classical"]:  # Try deep then classical configs
        try:
            default_path = f"configs/default/{model_type}.json"
            opt_space_path = f"configs/opt_space/{model_type}.json"

            with open(default_path, "r") as f:
                default_para_model = json.load(f)
            with open(opt_space_path, "r") as f:
                opt_space_model = json.load(f)

            model_config = default_para_model[model_type]
            config_found = True
            if config_type == "classical":
                classical_path = "configs/classical_configs.json"

                with classical_path.open("r") as f:
                    classical_configs = json.load(f)
                classical_configs.update(model_config)
                model_config = classical_configs
            elif config_type == "deep":
                deep_path = "configs/deep_configs.json"
                with open(deep_path, "r") as f:
                    deep_configs = json.load(f)
                deep_configs.update(model_config)
                model_config = deep_configs
            break  # Found config, no need to check other types
        except FileNotFoundError:
            continue  # Try next config type

    if not config_found:
        print(
            f"Warning: Config files for model '{model_type}' not found in TALENT's default/opt_space. HPO might be skipped."
        )

    model_config.update(args)

    namespace_config = Namespace(**{"config": model_config, **args, **model_config})
    return namespace_config, opt_space_model


# pip tabular foundation models run as plain sklearn estimators (no TALENT
# Method, no config, no HPO space) inside the same evaluate_talent_dataset flow.
NEW_TFM_SPECS = {
    "tabiclv2": dict(
        package="tabicl",
        min_version=(2, 1),
        model_version="tabicl-v2",
        checkpoint="tabicl-classifier-v2-20260212.ckpt",
        hf_repo="jingang/TabICL",
        n_estimators=8,
        softmax_temperature=0.9,
    ),
    "tabpfn3": dict(
        package="tabpfn",
        min_version=(8, 0),
        model_version="v3",
        checkpoint="tabpfn-v3-classifier-v3_default.ckpt",
        hf_repo="Prior-Labs/tabpfn_3",
        n_estimators=8,
        softmax_temperature=0.9,
    ),
    "tabfm": dict(
        package="tabfm",
        min_version=(1, 0),
        model_version="tabfm-v1.0.0",
        checkpoint="classification/model.safetensors",
        hf_repo="google/tabfm-1.0.0-pytorch",
        n_estimators=8,
        softmax_temperature=0.9,
        # float32 rather than the vendor's bfloat16: portable to every card in
        # the fleet, so a cell does not depend on where it was scheduled
        compute_dtype="float32",
    ),
    "exaone": dict(
        package="exaonetabular",
        min_version=(1, 0),
        model_version="exaone-tabular-v1",
        checkpoint="exaone-tabular-classifier-v1_default.safetensors",
        hf_repo="LG-AI-Research/EXAONE-Tabular",
        n_estimators=8,
        softmax_temperature=None,   # the model exposes no temperature knob
    ),
    "tabldm": dict(
        package="Xiaomi-TabLDM",   # import name is tabldm; the DISTRIBUTION differs
        min_version=(0, 1),
        model_version="tabldm-v1",
        checkpoint="checkpoints/clf_default.ckpt",
        hf_repo="occams/Xiaomi-TabLDM",
        n_estimators=8,
        softmax_temperature=1.0,   # vendor default is 0.9 (post-hoc sharpening);
                                   # pinned to 1.0 so the roster is comparable
    ),
    "tabpfn26": dict(
        package="tabpfn",
        min_version=(8, 0),
        model_version="v2.6",
        checkpoint="tabpfn-v2.6-classifier-v2.6_default.ckpt",
        hf_repo="Prior-Labs/tabpfn_2_6",
        n_estimators=8,
        softmax_temperature=0.9,
    ),
    "tabpfn25b": dict(
        package="tabpfn",
        min_version=(8, 0),
        model_version="v2.5",
        # the sibling checkpoint, not the release default; see the module note
        checkpoint="tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
        checkpoint_is_variant=True,
        hf_repo="Prior-Labs/tabpfn_2_5",
        n_estimators=8,
        softmax_temperature=0.9,
    ),
    "tabpfn25": dict(
        package="tabpfn",
        min_version=(8, 0),
        model_version="v2.5",
        checkpoint="tabpfn-v2.5-classifier-v2.5_default.ckpt",
        hf_repo="Prior-Labs/tabpfn_2_5",
        n_estimators=8,
        softmax_temperature=0.9,
    ),
}
NEW_TFM_MODELS = set(NEW_TFM_SPECS)
NEW_TFM_HPO_REASON = "no HPO space (foundation model)"

IN_CONTEXT_MODELS = {"tabpfn", "PFN-v2", "tabicl", "mitra"} | NEW_TFM_MODELS

# Models whose ensemble size / softmax temperature the CLI may override
# (--n-estimators / --softmax-temperature). tabpfn v1 has no temperature knob
# (it is loaded from the checkpoint); mitra has no ensemble.
N_ESTIMATORS_MODELS = {"tabpfn", "PFN-v2", "tabicl"} | NEW_TFM_MODELS
SOFTMAX_TEMPERATURE_MODELS = {"PFN-v2", "tabicl"} | {
    m for m in NEW_TFM_MODELS if NEW_TFM_SPECS[m]["softmax_temperature"] is not None
}

# What the vendored constructors pass when nothing is overridden
# (model/methods/{tabpfn,PFN_v2,tabicl}.py); recorded in meta for the paper.
VENDORED_TFM_DEFAULTS = {
    "tabpfn": dict(n_estimators=3, softmax_temperature=None, model_version="v1"),
    "PFN-v2": dict(n_estimators=4, softmax_temperature=0.9, model_version="v2"),
    "tabicl": dict(n_estimators=32, softmax_temperature=0.9, model_version="tabicl-v1.1"),
    "mitra": dict(n_estimators=1, softmax_temperature=None, model_version=None),
}


def uses_validation_split(model_name: str) -> bool:
    """Epoch-trained deep models select ``best-val`` and early-stop on ``val``;
    they are the only ones that need a held-out validation split."""
    return model_name in DEEP_MODELS and model_name not in IN_CONTEXT_MODELS


def _stratified_split(*arrays, test_size, seed, stratify):
    """train_test_split that falls back to a plain shuffle when some class has
    fewer than two members (``wine-quality-white`` has a 5-member class that
    breaks the 25 % HPO split on small seeds)."""
    _, counts = np.unique(stratify, return_counts=True)
    strat = stratify if counts.min() >= 2 else None
    return train_test_split(
        *arrays, test_size=test_size, random_state=seed, stratify=strat
    )


def _has_tunable_params(opt_space_model, model_name) -> bool:
    space = opt_space_model.get(model_name, {})
    return any(bool(group) for group in space.values())


def evaluate_talent_dataset(
    dataset_name,
    dataset_path,
    models_to_run,
    device,
    confidence_level=0.9,
    conformity_scores=("lac",),
    seed=42,
    n_trials=25,
    logger: Optional[logging.Logger] = None,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_error: Optional[Callable[[Dict[str, Any]], None]] = None,
    mock_run: bool = False,
    cache_dir: str = "cache",
    work_dir: str = "./",
    machine: str = "unknown",
    env_tag: str = "",
    git_sha: str = "",
    val_frac: float = 0.15,
    overwrite: bool = False,
    n_estimators: Optional[int] = None,
    softmax_temperature: Optional[float] = None,
    tag: Optional[str] = None,
    hpo_objective: str = "accuracy",
):
    """Evaluates models on a single TALENT dataset for one seed.

    ``hpo_objective`` selects what the Optuna study maximises for the tuned
    classifiers: ``accuracy`` (TALENT's validation accuracy, log-loss
    tie-break) or ``logloss`` (negative validation log-loss, accuracy
    tie-break; E-lltune). Recorded as ``meta["hpo_objective"]`` and in the
    ``hpo_objective`` column of every row; anything but ``accuracy`` must
    run under a ``tag`` (talent_benchmark.py enforces it).

    ``n_estimators`` / ``softmax_temperature`` override the ensemble size and
    the softmax temperature of the in-context models that have them
    (``N_ESTIMATORS_MODELS`` / ``SOFTMAX_TEMPERATURE_MODELS``); ``None`` leaves
    every constructor call exactly as it is. ``tag`` suffixes the cache cell
    (``<model>@<tag>``) so a sweep never collides with the base grid.

    One fit per model. The fitted model is asked for probabilities on the
    calibration and on the test split exactly once; those are written to
    ``cache_dir`` and read back, and every conformity score in
    ``conformity_scores`` is then computed from the cache. One result row per
    score is emitted through ``on_result``.
    """
    if isinstance(conformity_scores, str):
        conformity_scores = [conformity_scores]
    conformity_scores = list(conformity_scores)

    talent_args, _ = get_talent_args(
        dataset_name, dataset_path, "tabpfn", seed, n_trials, work_dir
    )

    train_val_data, test_data, info = get_dataset(
        talent_args.config["dataset"], talent_args.config["dataset_path"]
    )
    (N_trainval, C_trainval, y_trainval) = train_val_data
    (N_test, C_test, y_test) = test_data
    task_type = info["task_type"]
    n_classes = int(info.get("n_classes") or len(np.unique(y_trainval["train"])))

    N_pool = (
        np.concatenate([N_trainval["train"], N_trainval["val"]]) if N_trainval else None
    )
    C_pool = (
        np.concatenate([C_trainval["train"], C_trainval["val"]]) if C_trainval else None
    )
    y_pool = np.concatenate([y_trainval["train"], y_trainval["val"]])

    X_pool = concat_features(C_pool, N_pool)
    X_test = concat_features(
        C_test["test"] if C_test else None, N_test["test"] if N_test else None
    )
    y_test_raw = y_test["test"]

    # row positions into TALENT's on-disk arrays (train+val concatenated; test)
    pool_idx = np.arange(X_pool.shape[0])
    test_idx = np.arange(X_test.shape[0])

    rng = np.random.default_rng(seed)
    if mock_run:
        mock_pool_limit = min(500, X_pool.shape[0])
        if X_pool.shape[0] > mock_pool_limit:
            sample_idx = rng.choice(
                X_pool.shape[0], size=mock_pool_limit, replace=False
            )
            X_pool = X_pool[sample_idx]
            y_pool = y_pool[sample_idx]
            pool_idx = pool_idx[sample_idx]
        mock_test_limit = min(200, X_test.shape[0])
        if X_test.shape[0] > mock_test_limit:
            test_idx = rng.choice(X_test.shape[0], size=mock_test_limit, replace=False)
            X_test = X_test[test_idx]
            y_test_raw = y_test_raw[test_idx]

    # Same call as before, with the index vector riding along: identical split.
    X_train, X_calib, y_train, y_calib, _, idx_calib = train_test_split(
        X_pool, y_pool, pool_idx, test_size=0.3, random_state=seed, stratify=y_pool
    )

    n_cat_features = 0
    if C_trainval and "train" in C_trainval:
        n_cat_features = C_trainval["train"].shape[1]
    n_num_features = X_pool.shape[1] - n_cat_features

    log = logger or logging.getLogger("talent_benchmark")

    all_results = []
    errors = []
    for model_name in models_to_run:
        existing = cache_exists(cache_dir, dataset_name, task_type, model_name, seed, tag=tag)
        if existing and not overwrite:
            log.info(
                "SKIP (cached) %s on %s seed %d -> %s",
                model_name,
                dataset_name,
                seed,
                existing,
            )
            continue

        log.info("Training %s on %s", model_name, dataset_name)
        try:
            if n_estimators is not None and model_name not in N_ESTIMATORS_MODELS:
                raise ValueError(f"--n-estimators is not supported by {model_name}")
            if softmax_temperature is not None and model_name not in SOFTMAX_TEMPERATURE_MODELS:
                raise ValueError(f"--softmax-temperature is not supported by {model_name}")

            hpo_ran = False
            hpo_stats: Dict[str, Any] = {}
            t_hpo = 0.0
            is_regression = info["task_type"] == "regression"
            if model_name in NEW_TFM_MODELS:
                # pip foundation model: no TALENT config, no HPO space.
                current_model_args, opt_space_model, tuned_args = None, {}, None
                hpo_reason = NEW_TFM_HPO_REASON
                best_params = {}
                log.info("Skipping HPO for %s (%s)", model_name, hpo_reason)
                talent_method = PipTFMClassifier(
                    model_name,
                    device,
                    seed,
                    n_cat_features=n_cat_features,
                    n_estimators=n_estimators,
                    softmax_temperature=softmax_temperature,
                )
                model = talent_method
            else:
                current_model_args, opt_space_model = get_talent_args(
                    dataset_name, dataset_path, model_name, seed, n_trials, work_dir,
                    hpo_objective=hpo_objective,
                )
            if model_name in NEW_TFM_MODELS:
                pass
            elif mock_run:
                hpo_reason = "quick-test mode"
            elif model_name not in opt_space_model:
                hpo_reason = "no HPO space found in configs"
            elif not _has_tunable_params(opt_space_model, model_name):
                hpo_reason = "empty HPO space"
            else:
                hpo_reason = None

            if hpo_reason is None:
                log.info(
                    "Tuning %s with TALENT's HPO (objective %s, %d trials)",
                    model_name,
                    hpo_objective,
                    n_trials,
                )

                # Split the main training set for HPO, keeping the calibration set separate.
                X_hpo_train, X_hpo_val, y_hpo_train, y_hpo_val = _stratified_split(
                    X_train, y_train, test_size=0.25, seed=seed, stratify=y_train
                )

                # Re-split X into C and N for TALENT's tuner. TALENT's own
                # pipeline (model/lib/data.py: get_dataset / data_nan_process)
                # expects a *missing* block to be ``None``, never a dict of
                # ``None``s - the latter crashed every Optuna trial on the 65
                # datasets without categorical columns.
                C_hpo = (
                    {
                        "train": X_hpo_train[:, :n_cat_features],
                        "val": X_hpo_val[:, :n_cat_features],
                    }
                    if n_cat_features > 0
                    else None
                )
                N_hpo = (
                    {
                        "train": X_hpo_train[:, n_cat_features:],
                        "val": X_hpo_val[:, n_cat_features:],
                    }
                    if n_num_features > 0
                    else None
                )
                hpo_data = (N_hpo, C_hpo, {"train": y_hpo_train, "val": y_hpo_val})

                t0 = time.perf_counter()
                tuned_args = tune_hyper_parameters(
                    current_model_args, opt_space_model, hpo_data, info
                )
                t_hpo = time.perf_counter() - t0
                hpo_stats = dict(getattr(tuned_args, "hpo_stats", {}) or {})
                hpo_ran = hpo_stats.get("n_trials_ok", 0) > 0
                best_params = {}
                for group_key, params_in_group in opt_space_model[model_name].items():
                    if group_key in tuned_args.config:
                        for param_name in params_in_group:
                            best_params[f"{group_key}_{param_name}"] = (
                                tuned_args.config[group_key][param_name]
                            )
            elif model_name in NEW_TFM_MODELS:
                pass  # no TALENT args at all; logged above
            else:
                log.info("Skipping HPO for %s (%s)", model_name, hpo_reason)
                tuned_args = current_model_args
                best_params = {}
                # configs/deep_configs.json carries ``tune: true`` since
                # 8c43afb (2025-10-21); the in-context methods (tabpfn,
                # PFN-v2, tabicl, mitra, hyperfast, tabptm) all
                # ``assert args.tune != True`` in __init__, so a model whose
                # HPO is skipped must be told that no tuning is happening.
                tuned_args.tune = False

            def _is_empty_value(v):
                return v is None or v == {} or (hasattr(v, "__len__") and len(v) == 0)

            if tuned_args is not None:
                for key in list(tuned_args.config.keys()):
                    if key in current_model_args and _is_empty_value(
                        tuned_args.config[key]
                    ):
                        tuned_args.config[key] = vars(current_model_args)[key]

                # Sweep overrides for the vendored in-context constructors
                # (model/methods/{tabpfn,PFN_v2,tabicl}.py read these through
                # getattr(..., None): unset -> the literal defaults, unchanged).
                tuned_args.n_estimators_override = n_estimators
                tuned_args.softmax_temperature_override = softmax_temperature

                talent_method = get_method(tuned_args.model_type)(tuned_args, is_regression)

                model = TalentScikitWrapper(talent_method, info, n_cat_features)

            # Held-out validation split for epoch-trained deep models only.
            use_val = uses_validation_split(model_name) and val_frac > 0
            if use_val:
                X_fit, X_val, y_fit, y_val = _stratified_split(
                    X_train, y_train, test_size=val_frac, seed=seed, stratify=y_train
                )
                log.info(
                    "Fitting %s on %d rows, early-stopping on %d held-out rows",
                    model_name,
                    len(y_fit),
                    len(y_val),
                )
            else:
                X_fit, y_fit, X_val, y_val = X_train, y_train, None, None
                log.info("Fitting %s on the full training data", model_name)

            t0 = time.perf_counter()
            model.fit(X_fit, y_fit, X_val, y_val)
            t_fit = time.perf_counter() - t0

            # Labels are stored *encoded* (0..K-1 = column order of p_*); the
            # raw label values go to meta.class_labels. TALENT labels are not
            # always 0..K-1 (satimage: {1,2,3,4,5,7}); the old path passed raw
            # y_calib to MAPIE, which re-encoded it through model.classes_.
            y_test_enc = talent_method.label_encoder.transform(y_test_raw)
            y_cal_enc = talent_method.label_encoder.transform(y_calib)
            class_labels = [str(c) for c in model.classes_]
            classes_pos = np.arange(len(class_labels))

            # One inference on each split; MAPIE never touches the model again.
            t0 = time.perf_counter()
            p_cal = np.asarray(model.predict_proba(X_calib), dtype=np.float64)
            t_pred_cal = time.perf_counter() - t0
            t0 = time.perf_counter()
            p_test = np.asarray(model.predict_proba(X_test), dtype=np.float64)
            t_pred_test = time.perf_counter() - t0

            # A constant classifier (every probability row identical: AUC = 0.500,
            # no empty sets, trivial q-hat) must never be written to the cache -
            # 62/1 110 XGBoost journal cells were (A1 audit, 2026-08-30). Raise so
            # the cell lands in __errors.csv and the grid is rerun, not tabulated.
            if task_type != "regression":
                const_test = is_constant_prediction(p_test, argmax=False)
                const_cal = is_constant_prediction(p_cal, argmax=False)
                if const_test or const_cal:
                    raise RuntimeError(
                        f"constant classifier: {model_name} on {dataset_name} seed {seed} "
                        f"predicts one identical probability row on "
                        f"{'test' if const_test else ''}{'+' if const_test and const_cal else ''}"
                        f"{'cal' if const_cal else ''} "
                        f"(p_test[0]={np.round(p_test[0], 6).tolist()}, "
                        f"hpo_constant_trials={int(hpo_stats.get('n_trials_constant', 0))}, "
                        f"best_params={best_params})"
                    )

            trlog = getattr(talent_method, "trlog", {}) or {}
            best_epoch = trlog.get("best_epoch") if use_val else None
            max_epoch = getattr(tuned_args, "max_epoch", None) if use_val else None
            ens = ensemble_info(model_name, talent_method)
            if ens.get("softmax_temperature") is None and tuned_args is not None:
                ens["softmax_temperature"] = (tuned_args.config.get("model") or {}).get(
                    "softmax_temperature"
                )

            meta = dict(
                dataset=dataset_name,
                task_type=task_type,
                n_classes=n_classes,
                class_labels=class_labels,
                model=model_name,
                seed=int(seed),
                mapie_random_state=int(seed),
                confidence_level=confidence_level,
                conformity_scores=conformity_scores,
                machine=machine,
                env_tag=env_tag,
                git_sha=git_sha,
                mock_run=bool(mock_run),
                n_train=int(len(y_train)),
                n_fit=int(len(y_fit)),
                n_val=int(0 if y_val is None else len(y_val)),
                val_frac=float(val_frac if use_val else 0.0),
                n_cal=int(len(y_calib)),
                n_test=int(len(y_test_enc)),
                n_cat_features=int(n_cat_features),
                n_num_features=int(n_num_features),
                hpo_ran=bool(hpo_ran),
                hpo_skipped_reason=hpo_reason,
                n_trials=int(hpo_stats.get("n_trials", n_trials if hpo_ran else 0)),
                n_trials_ok=int(hpo_stats.get("n_trials_ok", 0)),
                n_trials_failed=int(hpo_stats.get("n_trials_failed", 0)),
                hpo_best_value=hpo_stats.get("best_value"),
                hpo_best_val_acc=hpo_stats.get("best_val_acc"),
                hpo_best_val_logloss=hpo_stats.get("best_val_logloss"),
                hpo_objective=hpo_stats.get("objective"),
                hpo_objective_formula=hpo_stats.get("objective_formula"),
                hpo_constant_trials=int(hpo_stats.get("n_trials_constant", 0)),
                hpo_argmax_constant_trials=int(hpo_stats.get("n_trials_argmax_constant", 0)),
                best_params=best_params,
                best_epoch=best_epoch,
                max_epoch=max_epoch,
                best_val_res=trlog.get("best_res") if use_val else None,
                t_hpo_s=t_hpo,
                t_fit_s=t_fit,
                t_pred_cal_s=t_pred_cal,
                t_pred_test_s=t_pred_test,
                tag=tag or "",
                n_estimators=ens.get("n_estimators"),
                n_estimators_effective=ens.get("n_estimators_effective"),
                n_estimators_override=n_estimators,
                softmax_temperature=ens.get("softmax_temperature"),
                softmax_temperature_override=softmax_temperature,
                average_logits=ens.get("average_logits"),
                average_before_softmax=ens.get("average_before_softmax"),
                auto_scale_n_estimators=ens.get("auto_scale_n_estimators"),
                model_version=ens.get("model_version"),
                checkpoint=ens.get("checkpoint"),
                checkpoint_path=ens.get("checkpoint_path"),
                package=ens.get("package"),
                package_version=ens.get("package_version"),
                categorical_features_indices=ens.get("categorical_features_indices"),
                support_subsampled=getattr(talent_method, "support_subsampled", None),
                versions=package_versions(),
                device=str(device),
            )

            # Score once per conformity score from the in-memory arrays to get
            # q-hat into the file; the CSV rows below are recomputed from the
            # file as read back, so what the paper uses is what is on disk.
            qhat = {}
            effective_scores, skipped_scores = [], {}
            for score in conformity_scores:
                try:
                    _, q = conformal_from_arrays(
                        p_cal, y_cal_enc, p_test, classes_pos, score, confidence_level, seed
                    )
                except Exception as exc:  # noqa: BLE001
                    # A score that cannot be built must not lose the cell: the
                    # probabilities are already computed and every other score is
                    # a function of them. Known cases: MAPIE 1.0.x "Invalid
                    # conformity score for binary target. The only valid score is
                    # 'lac'." (aps/raps/top_k on K = 2) and RAPS's stratified
                    # size_raps split on a calibration class with a single member
                    # (wine-quality-white, found 2026-08-30 on every model).
                    skipped_scores[score] = f"{type(exc).__name__}: {exc}"
                    log.warning("SKIP score %s for %s on %s: %s", score, model_name, dataset_name, exc)
                    continue
                effective_scores.append(score)
                if q is not None:
                    qhat[score] = q
            if not effective_scores:
                raise RuntimeError(
                    f"no conformity score usable on {dataset_name}: {skipped_scores}"
                )
            meta["conformity_scores"] = effective_scores
            meta["conformity_scores_skipped"] = skipped_scores

            npz_path = cache_path(
                cache_dir, dataset_name, task_type, model_name, seed, machine, tag=tag
            )
            write_cache(
                npz_path,
                p_cal=p_cal,
                y_cal=y_cal_enc,
                p_test=p_test,
                y_test=y_test_enc,
                idx_cal=idx_calib,
                idx_test=test_idx,
                classes=classes_pos,
                X_cal=X_calib,
                X_test=X_test,
                meta=meta,
                qhat=qhat,
            )
            cached = read_cache(npz_path)
            append_manifest(
                os.path.join(cache_dir, "manifest.csv"),
                {
                    **meta,
                    "cache_file": os.path.basename(npz_path),
                    "x_cal_sha1": cached["x_cal_sha1"],
                    "x_test_sha1": cached["x_test_sha1"],
                    "written_at": meta.get("written_at", cached["meta"]["written_at"]),
                },
            )

            y_prob = cached["p_test"]
            y_pred = np.argmax(y_prob, axis=1)
            for score in effective_scores:
                log.info("Conformalizing %s with %s from the cache", model_name, score)
                y_pred_set, _ = conformal_from_cache(cached, score)
                test_metrics = evaluate_classification(
                    y_pred, cached["y_test"], y_pred_set, y_prob
                )

                prefixed_metrics = {
                    f"{model_name}_{k}": v for k, v in test_metrics.items()
                }
                prefixed_params = {
                    f"{model_name}_best_{k}": v for k, v in best_params.items()
                }

                result = {
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "n_classes": n_classes,
                    "seed": seed,
                    "model": model_name,
                    "tag": tag or "",
                    "confidence_level": confidence_level,
                    "conformity_score": score,
                    **test_metrics,
                    "machine": machine,
                    "env_tag": env_tag,
                    "git_sha": git_sha,
                    "cache_path": os.path.basename(npz_path),
                    "hpo_ran": bool(hpo_ran),
                    "hpo_objective": meta["hpo_objective"],
                    "n_trials": meta["n_trials"],
                    "n_trials_ok": meta["n_trials_ok"],
                    "hpo_constant_trials": meta["hpo_constant_trials"],
                    "t_hpo_s": t_hpo,
                    "t_fit_s": t_fit,
                    "t_pred_s": t_pred_cal + t_pred_test,
                    "t_pred_cal_s": t_pred_cal,
                    "t_pred_test_s": t_pred_test,
                    "best_epoch": best_epoch,
                    "max_epoch": max_epoch,
                    "val_frac": meta["val_frac"],
                    "n_train": meta["n_train"],
                    "n_cal": meta["n_cal"],
                    "n_test": meta["n_test"],
                    "n_estimators": meta["n_estimators"],
                    "softmax_temperature": meta["softmax_temperature"],
                    "model_version": meta["model_version"],
                    "wandb_log": {
                        **prefixed_metrics,
                        **prefixed_params,
                        "dataset": dataset_name,
                        "seed": seed,
                        "conformity_score": score,
                    },
                }
                all_results.append(result)
                if on_result:
                    on_result(result)
            log.info(
                "Successfully evaluated %s on %s (hpo_ran=%s, t_hpo=%.1fs, t_fit=%.1fs, best_epoch=%s)",
                model_name,
                dataset_name,
                hpo_ran,
                t_hpo,
                t_fit,
                best_epoch,
            )
        except Exception as exc:  # noqa: BLE001
            error_record = {
                "dataset": dataset_name,
                "task_type": task_type,
                "seed": seed,
                "model": model_name,
                "tag": tag or "",
                "conformity_score": "|".join(conformity_scores),
                "error": repr(exc),
            }
            errors.append(error_record)
            if on_error:
                on_error(error_record)
            log.exception(
                "Error while evaluating %s on %s. Continuing with next model.",
                model_name,
                dataset_name,
            )
            continue

    return all_results, errors


def _evaluate_dataset_splits(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    models_to_run,
    device,
    dataset_identifier,
    score="lac",
    confidence_level=0.9,
    seed=42,
    n_trials=25,
):
    """
    Core evaluation logic for a given dataset split.

    Args:
        X_train, y_train: Training data and labels.
        X_val, y_val: Calibration/validation data and labels.
        X_test, y_test: Test data and labels.
        models_to_run: List of model names to evaluate.
        device: The device to run on ('cpu', 'cuda', etc.).
        dataset_identifier: A string or int to identify the dataset in results.
        score: The conformity score for MAPIE.
        confidence_level: The confidence level for prediction sets.
        seed: The random seed.
        n_trials: Number of HPO trials.

    Returns:
        A list of dictionaries containing the results for each model.
    """
    vectorizer = TableVectorizer(drop_if_constant=True, sparse_threshold=0.1)
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    X_train.columns = [clean_col(col) for col in X_train.columns]
    X_val.columns = [clean_col(col) for col in X_val.columns]
    X_test.columns = [clean_col(col) for col in X_test.columns]

    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    results = []

    devices = None
    task_type = None
    if "cuda" in str(device):
        devices = str(device).split(":")[1] if ":" in str(device) else "0"
        task_type = "GPU"

    for model_name in tqdm(models_to_run, desc=f"Models on {dataset_identifier}"):
        print(f"--- Training {model_name} ---")
        model = ALL_MODELS[model_name]
        best_params = {}

        if model_name in HPO_SEARCH_SPACES:
            print(f"Tuning {model_name} with Optuna...")
            model_instantiator = MODEL_INSTANTIATORS[model_name](
                seed, task_type, devices
            )
            best_params = tune_model_with_optuna(
                model_class=model_instantiator,
                trial_params=HPO_SEARCH_SPACES[model_name],
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=n_trials,
            )
            print(f"Best params for {model_name}: {best_params}")
            model = model_instantiator(**best_params)

        if "verbose" in model.get_params():
            model.set_params(verbose=-1)
        if "verbosity" in model.get_params():
            model.set_params(verbosity=0)

        model.fit(X_train, y_train)

        mapie_clf = SplitConformalClassifier(
            estimator=model,
            confidence_level=confidence_level,
            prefit=True,
            conformity_score=score,
            random_state=seed,
        )
        mapie_clf.conformalize(X_val, y_val)

        y_prob = model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)
        _, y_pred_set = mapie_clf.predict_set(X_test)
        test_metrics = evaluate_classification(y_pred, y_test, y_pred_set, y_prob)

        prefixed_metrics = {f"{model_name}_{k}": v for k, v in test_metrics.items()}
        prefixed_params = {f"{model_name}_best_{k}": v for k, v in best_params.items()}

        results.append(
            {
                "dataset": dataset_identifier,
                "seed": seed,
                "model": model_name,
                "confidence_level": confidence_level,
                **test_metrics,
                "wandb_log": {**prefixed_metrics, **prefixed_params},
            }
        )

    return results


def evaluate_on_openml(
    dataset_id,
    device,
    score="lac",
    confidence_level=0.9,
    seed=42,
    wandb_config=None,
):
    df: Bunch = fetch_openml(data_id=dataset_id, as_frame="auto")  # type: ignore
    X = df.data

    if isinstance(X, csr_matrix):
        X = pd.DataFrame(X.toarray(), columns=df.feature_names)

    float64_cols = X.select_dtypes(np.float64).columns
    X[float64_cols] = X[float64_cols].astype(np.float32)

    cols_to_drop = X.columns[X.nunique() <= 1]
    if not cols_to_drop.empty:
        X = X.drop(columns=cols_to_drop)  # type: ignore

    if df.target.dtype.name == "category":
        y = df.target.cat.codes.to_numpy(dtype=int)
    elif df.target.dtype.name == "object":
        y = (df.target == "Yes").to_numpy(dtype=int)
    else:
        y = pd.factorize(df.target)[0]

    X_train, X_calib, X_test, y_train, y_calib, y_test = train_conformalize_test_split(
        X,
        y,
        train_size=0.5,
        conformalize_size=0.3,
        test_size=0.2,
        random_state=seed,  # type: ignore
    )

    # Get all model names, as evaluate_on_openml runs on all of them.
    models_to_run = list(ALL_MODELS.keys())

    return _evaluate_dataset_splits(
        X_train,
        y_train,
        X_calib,
        y_calib,
        X_test,
        y_test,
        models_to_run,
        device,
        dataset_id,
        score,
        confidence_level,
        seed,
    )


# --- Define Hyperparameter Search Spaces for Optuna ---
HPO_SEARCH_SPACES = {
    "LightGBM": {
        "n_estimators": ("int", {"low": 50, "high": 500}),
        "learning_rate": ("float", {"low": 0.005, "high": 0.1, "log": True}),
        "num_leaves": ("int", {"low": 20, "high": 150}),
        "max_depth": ("int", {"low": 3, "high": 12}),
        "reg_alpha": ("float", {"low": 1e-3, "high": 1.0, "log": True}),
        "reg_lambda": ("float", {"low": 1e-3, "high": 1.0, "log": True}),
    },
    "CatBoost": {
        "iterations": ("int", {"low": 100, "high": 600}),
        "depth": ("int", {"low": 4, "high": 10}),
        "learning_rate": ("float", {"low": 0.005, "high": 0.1, "log": True}),
    },
    "XGBoost": {
        "n_estimators": ("int", {"low": 50, "high": 500}),
        "max_depth": ("int", {"low": 3, "high": 9}),
        "learning_rate": ("float", {"low": 0.005, "high": 0.1, "log": True}),
        "subsample": ("float", {"low": 0.6, "high": 1.0}),
        "colsample_bytree": ("float", {"low": 0.6, "high": 1.0}),
    },
}

# --- Define Model Instantiators ---
# We use lambdas to pass fixed parameters like random_state and device
MODEL_INSTANTIATORS = {
    "LightGBM": lambda seed, task_type, devices: (
        lambda **kwargs: LGBMClassifier(
            random_state=seed, force_col_wise=True, **kwargs
        )
    ),
    "CatBoost": lambda seed, task_type, devices: (
        lambda **kwargs: CatBoostClassifier(
            task_type=task_type,
            devices=devices,
            random_state=seed,
            allow_writing_files=False,
            **kwargs,
        )
    ),
    "XGBoost": lambda seed, task_type, devices: (
        lambda **kwargs: XGBClassifier(  # type: ignore
            device="cpu" if devices.type in ["mps", "cpu"] else devices.type,
            eval_metric="logloss",
            random_state=seed,
            **kwargs,
        )
    ),
}

# --- Main Model Dictionary ---
ALL_MODELS = {
    "LightGBM": LGBMClassifier(verbose=-100),
    "CatBoost": CatBoostClassifier(verbose=False, allow_writing_files=False),
    "XGBoost": XGBClassifier(eval_metric="logloss", verbosity=0),
    "LogisticRegression": make_pipeline(
        SimpleImputer(strategy="mean"),
        LogisticRegression(max_iter=1000),
    ),
    "TabICL": TabICLClassifier(n_estimators=1),
    "TabPFN": TabPFNClassifier(n_estimators=1),
    "TabM": "TabM",  # Placeholder, will be instantiated in the loop
}
