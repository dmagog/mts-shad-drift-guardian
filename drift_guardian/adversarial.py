"""Adversarial Validation: классификатор учится отличать эталон от текущего батча.

Идея: объединяем обе выборки, помечаем строки метками «эталон/продакшн» и учим
модель их различать. Если ROC-AUC существенно выше 0.5 — совместное распределение
признаков изменилось, а важности признаков показывают, какие колонки изменились
сильнее всего. Метрика устойчива к типам данных и ловит многомерные сдвиги,
которые не видны в поколоночных тестах.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import DriftConfig
from .contracts import AdversarialReport

try:  # LightGBM — основной бэкенд (по ТЗ); fallback на случай проблем со сборкой.
    from lightgbm import LGBMClassifier

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import RandomForestClassifier

    _HAS_LIGHTGBM = False


def _auto_n_jobs(config: DriftConfig, n_rows: int, n_cols: int) -> int:
    """1 поток на небольших данных (накладные расходы), 4 — на матрицах от миллиона ячеек."""
    if config.adversarial_n_jobs is not None:
        return config.adversarial_n_jobs
    return 4 if n_rows * n_cols >= 1_000_000 else 1


def _row_hashes(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


def _make_model(random_state: int, n_jobs: int, n_cols: int = 0):
    if _HAS_LIGHTGBM:
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            importance_type="gain",
            colsample_bytree=0.5 if n_cols > 50 else 1.0,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
        return model, "lightgbm"
    model = RandomForestClassifier(  # pragma: no cover
        n_estimators=200, max_depth=8, random_state=random_state, n_jobs=n_jobs
    )
    return model, "sklearn-random-forest"


def _encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Категории — в целочисленные коды (NaN -> -1); числовые колонки — как есть."""
    out: dict[str, np.ndarray | pd.Series] = {}
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            out[col] = pd.to_numeric(series, errors="coerce")
        else:
            codes, _ = pd.factorize(series, use_na_sentinel=True)
            out[col] = codes
    return pd.DataFrame(out, index=frame.index)


def adversarial_validation(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    config: DriftConfig,
) -> AdversarialReport | None:
    """Обучает классификатор «эталон против текущего батча» и оценивает ROC-AUC.

    Возвращает ``None``, если данных слишком мало для осмысленной оценки.
    """
    th = config.thresholds
    if not columns:
        return None

    ref = reference[columns]
    cur = current[columns]

    # Строки-двойники: точные совпадения между эталоном и батчем (перекрывающиеся окна,
    # общие клиенты) неинформативны для классификатора, но через кросс-валидацию
    # «подсказывают» ему метку двойника — на идентичных выборках это давало AUC ≈ 0.77.
    # Такие строки исключаются из обучения, их доля попадает в отчёт.
    ref_hash, cur_hash = _row_hashes(ref), _row_hashes(cur)
    common = np.intersect1d(ref_hash, cur_hash)
    overlap_share = float(np.isin(cur_hash, common).mean()) if len(cur) else 0.0
    note = ""
    if len(common):
        ref = ref.loc[~np.isin(ref_hash, common)]
        cur = cur.loc[~np.isin(cur_hash, common)]
        note = (
            f"{overlap_share:.0%} строк батча точно совпадают со строками эталона "
            "и исключены из adversarial-валидации."
        )

    cap = max(config.adversarial_max_rows // 2, 1)
    if len(ref) > cap:
        ref = ref.sample(cap, random_state=config.random_state)
    if len(cur) > cap:
        cur = cur.sample(cap, random_state=config.random_state)
    if min(len(ref), len(cur)) < 50:
        if overlap_share > 0:
            return AdversarialReport(
                roc_auc=0.5, severity="ok", backend="—", n_rows_used=0,
                overlap_share=round(overlap_share, 4),
                note=(
                    f"{overlap_share:.0%} строк батча совпадают с эталоном; после их исключения "
                    "данных для adversarial-валидации не осталось — выборки практически идентичны."
                ),
            )
        return None

    X = _encode(pd.concat([ref, cur], ignore_index=True))
    y = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))])

    n_jobs = _auto_n_jobs(config, len(X), len(columns))
    model, backend = _make_model(config.random_state, n_jobs, len(columns))
    if backend != "lightgbm":  # pragma: no cover — RandomForest не понимает NaN
        X = X.fillna(-999_999.0)

    n_splits = 3 if min(len(ref), len(cur)) >= 150 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    auc = max(auc, 1.0 - auc)  # различимость в любую сторону

    model.fit(X, y)
    importances = np.asarray(model.feature_importances_, dtype=float)
    total = float(importances.sum()) or 1.0
    order = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": str(X.columns[i]), "importance": round(float(importances[i] / total), 4)}
        for i in order
        if importances[i] > 0
    ]

    if auc >= th.adversarial_auc_critical:
        severity = "critical"
    elif auc >= th.adversarial_auc_warning:
        severity = "warning"
    else:
        severity = "ok"

    return AdversarialReport(
        roc_auc=round(auc, 4),
        severity=severity,
        backend=backend,
        n_rows_used=int(len(X)),
        top_features=top_features,
        overlap_share=round(overlap_share, 4),
        note=note,
    )
