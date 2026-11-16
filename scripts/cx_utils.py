"""
Utilidades del experimento ChronosX en M5 (datos, validacion, reanudacion, metricas por serie y bootstrap).
Las usan run_chronosx.py y orchestrate.py (en esta misma carpeta).
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split

CONFIG_PATH = str(Path(__file__).resolve().parent / "configs" / "datasets.yaml")


# ------------------------------------------------------------------ datos
def load_cfg(dataset: str, path: str = CONFIG_PATH) -> dict:
    with open(path) as fp:
        for c in yaml.safe_load(fp):
            if c["name"] == dataset:
                return c
    raise ValueError(f"'{dataset}' no esta en {path}")


class LazyUnivariate:
    """Misma salida que to_gluonts_univariate (una serie objetivo por fila), pero perezoso."""

    def __init__(self, hf_dataset, series_field: str, covariates_fields):
        assert isinstance(series_field, str), "solo se soporta un campo objetivo por dataset"
        self.series_field = series_field
        self.cov_fields = list(covariates_fields or [])
        cols = ["timestamp", series_field] + self.cov_fields
        self.ds = hf_dataset.select_columns(cols)  # evita leer las columnas estaticas (strings)
        self.ds.set_format("numpy")
        self.freq = pd.DatetimeIndex(self.ds[0]["timestamp"]).to_period()[0].freqstr

    def __len__(self):
        return len(self.ds)

    def _entry(self, row):
        entry = {
            "start": pd.Period(row["timestamp"][0], freq=self.freq),
            "target": row[self.series_field],
        }
        if self.cov_fields:
            cov = np.array([row[f] for f in self.cov_fields])
            is_nan = np.isnan(cov)
            cov[is_nan] = -1
            cov = np.vstack([cov, is_nan]).astype(np.float32)
            entry[FieldName.FEAT_DYNAMIC_REAL] = cov
        return entry

    def __iter__(self):
        for row in self.ds:
            yield self._entry(row)


def _get_hf_dataset(cfg: dict, num_series: int, seed: int):
    import os

    name = cfg["name"]
    local = os.environ.get("CX_LOCAL_DATASET")  # dataset HF guardado en disco (pruebas / cache propia)
    if local:
        import datasets

        ds = datasets.load_from_disk(local)
        if num_series and num_series < len(ds):
            ds = ds.select(np.sort(np.random.default_rng(seed).choice(len(ds), num_series, replace=False)).tolist())
        return ds
    if name != "m5_with_covariates":
        raise ValueError(f"Este repositorio solo soporta 'm5_with_covariates' (recibido: '{name}')")
    from chronosx.utils.m5_lean import build_m5  # version de bajo consumo de RAM

    # M5 se descarga y se arma en el directorio actual (./m5 y ./m5_lean_*), o en $CX_DATA_DIR
    return build_m5(num_series=num_series, seed=seed, directory=os.environ.get("CX_DATA_DIR", "."))


def load_data(dataset: str, num_series: int = 0, seed: int = 0, num_rolls=None):
    """Devuelve dict con: cfg, H, num_covariates, hf (dataset HF), train_dataset, test_data."""
    cfg = load_cfg(dataset)
    H = cfg["prediction_length"]
    rolls = num_rolls or cfg["num_rolls"]
    hf = _get_hf_dataset(cfg, num_series, seed)
    lazy = LazyUnivariate(hf, cfg["series_fields"], cfg.get("covariates_fields"))
    # con k ventanas de test consecutivas, gluonts exige offset <= -H*k (para k=1 es el offset del yaml)
    offset = cfg["offset"] * rolls
    train_dataset, test_template = split(lazy, offset=offset)
    test_data = test_template.generate_instances(H, windows=rolls)
    return {
        "cfg": cfg,
        "H": H,
        "offset": offset,
        "num_rolls": rolls,
        "num_covariates": 2 * len(cfg.get("covariates_fields") or []),
        "hf": hf,
        "n_series": len(lazy),
        "train_dataset": train_dataset,
        "test_data": test_data,
    }


def validation_entries(data: dict, val_series: int, seed: int):
    """Lista (en RAM) de series de validacion: subconjunto ALEATORIO y reproducible.

    (La version anterior tomaba las primeras N series, que en M5 son todas de FOODS_1.)
    """
    from functools import partial

    from chronosx.utils.utils import has_enough_observations
    from gluonts.itertools import Filter

    cfg, H, hf = data["cfg"], data["H"], data["hf"]
    n = len(hf)
    if val_series and val_series < n:
        idx = np.sort(np.random.default_rng(seed).choice(n, size=val_series, replace=False))
        hf = hf.select(idx.tolist())
    lazy = LazyUnivariate(hf, cfg["series_fields"], cfg.get("covariates_fields"))
    train_part, _ = split(lazy, offset=data["offset"])
    filt = Filter(partial(has_enough_observations, min_length=1 + 2 * H, max_missing_prop=0.5), train_part)
    return list(filt)


def training_stream(data: dict):
    """Serie de entrenamiento = ultimas H observaciones quedan fuera (como el script del repo)."""
    from functools import partial

    from chronosx.utils.utils import has_enough_observations
    from gluonts.itertools import Filter

    H = data["H"]
    filt = Filter(
        partial(has_enough_observations, min_length=1 + 2 * H, max_missing_prop=0.5),
        data["train_dataset"],
    )
    train_split, _ = split(filt, offset=-H)
    return train_split


def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# ------------------------------------------------------------------ reanudacion
def checkpoint_problem(path):
    """Devuelve None si el checkpoint esta completo, o un texto con lo que le falta.

    Drive sube los archivos con retraso: si la sesion se corta, un checkpoint puede quedar con
    optimizer.pt y trainer_state.json pero SIN los pesos (o con ellos truncados). Reanudar de ahi falla.
    """
    import struct

    p = Path(path)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        f = p / name
        if not f.exists() or f.stat().st_size == 0:
            return f"falta {name}"
    st, pt = p / "model.safetensors", p / "pytorch_model.bin"
    if st.exists():
        size = st.stat().st_size
        try:
            with open(st, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            end = max([v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"] or [0])
        except Exception as e:  # cabecera ilegible = archivo cortado
            return f"model.safetensors ilegible ({type(e).__name__})"
        if size != 8 + n + end:
            return f"model.safetensors truncado ({size} bytes, esperaba {8 + n + end})"
        return None
    if pt.exists() and pt.stat().st_size > 0:
        return None
    return "faltan los pesos del modelo (model.safetensors)"


def latest_complete_checkpoint(output_dir):
    """(ruta o None, lista de (ruta, motivo) de los checkpoints descartados), del mas nuevo al mas viejo."""
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]), reverse=True
    )
    skipped = []
    for c in ckpts:
        why = checkpoint_problem(c)
        if why is None:
            return c, skipped
        skipped.append((c, why))
    return None, skipped


def enable_trainer_resume():
    """Parchea Trainer.train para reanudar desde el ultimo checkpoint-* COMPLETO de output_dir."""
    from transformers import Trainer

    if getattr(Trainer, "_cx_resume_patched", False):
        return
    original = Trainer.train

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if resume_from_checkpoint is None:
            best, skipped = latest_complete_checkpoint(self.args.output_dir)
            for c, why in skipped:
                print(f"[resume] IGNORO {c.name}: {why}", flush=True)
            if best is not None:
                print(f"[resume] reanudando desde {best}", flush=True)
                resume_from_checkpoint = str(best)
            elif skipped:
                print("[resume] ningun checkpoint esta completo: se entrena desde el paso 0", flush=True)
        return original(self, resume_from_checkpoint, *args, **kwargs)

    Trainer.train = train
    Trainer._cx_resume_patched = True


def enable_light_eval():
    """Evita que la validacion agote la RAM.

    Durante Trainer.evaluate() el Trainer acumula en RAM TODAS las salidas del modelo. Con el bloque OIB,
    ChronosX devuelve ademas los hidden states del encoder y del decoder (varios MB por serie), y
    compute_metrics solo usa los logits. Aqui se descartan las demas salidas antes de acumularlas:
    la perdida de validacion es la misma, pero la memoria deja de crecer con ~9 MB por serie.
    """
    from transformers import Trainer

    if getattr(Trainer, "_cx_light_eval", False):
        return
    original_init = Trainer.__init__

    def keep_only_logits(logits, labels):
        first = logits[0] if isinstance(logits, (tuple, list)) else logits
        return (first,)

    def init(self, *args, **kwargs):
        if kwargs.get("preprocess_logits_for_metrics") is None:
            kwargs["preprocess_logits_for_metrics"] = keep_only_logits
        original_init(self, *args, **kwargs)

    Trainer.__init__ = init
    Trainer._cx_light_eval = True


# ------------------------------------------------------------------ metricas
QUANTILES = np.arange(0.05, 1, 0.05).round(2).tolist()


def item_metrics(forecasts, test_data, quantiles=QUANTILES, batch_size=500) -> dict:
    """Metricas POR SERIE con gluonts (axis=1 = se agrega solo en el tiempo).

    mase[n]      : MASE de la mediana
    sql[n, Q]    : suma en el tiempo de la quantile loss, por cuantil
    abs[n]       : suma de |y| en el horizonte
    El WQL agregado de gluonts es mean_q( sum_i sql[i,q] / sum_i abs[i] ).
    batch_size chico a proposito: gluonts mantiene en RAM el lote de series con sus covariables.
    """
    from gluonts.ev.metrics import MASE, SumQuantileLoss, sum_absolute_label
    from gluonts.model.evaluation import evaluate_forecasts_raw

    metrics = [MASE()] + [SumQuantileLoss(q=q) for q in quantiles] + [sum_absolute_label]
    raw = evaluate_forecasts_raw(
        forecasts, test_data=test_data, metrics=metrics, axis=1, batch_size=batch_size
    )

    def arr(key):
        return np.ma.filled(np.ma.asarray(raw[key], dtype=float), np.nan).reshape(-1)

    return {
        "mase": arr("MASE[0.5]"),
        "sql": np.stack([arr(f"sum_quantile_loss[{q}]") for q in quantiles], axis=1),
        "abs": arr("sum_absolute_label"),
    }


def aggregate(m: dict) -> dict:
    ok = np.isfinite(m["mase"])
    return {
        "MASE[0.5]": float(np.mean(m["mase"][ok])),
        "mean_weighted_sum_quantile_loss": float(np.mean(np.nansum(m["sql"], axis=0) / np.nansum(m["abs"]))),
        "n_items": int(len(m["mase"])),
        "n_items_mase_invalid": int((~ok).sum()),
    }


def paired_bootstrap(base: dict, new: dict, n_boot: int = 1000, seed: int = 0, min_items: int = 30):
    """Bootstrap emparejado sobre series: IC95% de la diferencia (new - base) en MASE y WQL.

    Devuelve None si hay menos de `min_items` series (p.ej. datasets de una sola serie).
    """
    n = len(base["mase"])
    if n < min_items:
        return None
    rng = np.random.default_rng(seed)
    ok = np.isfinite(base["mase"]) & np.isfinite(new["mase"])
    d_mase = (new["mase"] - base["mase"])[ok]
    b_mase = base["mase"][ok]
    n_mase = new["mase"][ok]

    out = {"mase_diff": [], "mase_rel": [], "wql_diff": [], "wql_rel": []}
    chunk = 50
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=size).astype(np.float64)  # (size, n)
        c_ok = counts[:, ok]
        w = c_ok.sum(1)
        mb = (c_ok @ b_mase) / w
        mn = (c_ok @ n_mase) / w
        out["mase_diff"].append(mn - mb)
        out["mase_rel"].append((mn - mb) / mb)
        den_b = counts @ base["abs"]
        den_n = counts @ new["abs"]
        wb = np.mean((counts @ np.nan_to_num(base["sql"])) / den_b[:, None], axis=1)
        wn = np.mean((counts @ np.nan_to_num(new["sql"])) / den_n[:, None], axis=1)
        out["wql_diff"].append(wn - wb)
        out["wql_rel"].append((wn - wb) / wb)
    out = {k: np.concatenate(v) for k, v in out.items()}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {
        "n_boot": n_boot,
        "mase_diff_ci95": ci(out["mase_diff"]),
        "mase_rel_pct_ci95": [100 * v for v in ci(out["mase_rel"])],
        "wql_diff_ci95": ci(out["wql_diff"]),
        "wql_rel_pct_ci95": [100 * v for v in ci(out["wql_rel"])],
        "prob_chronosx_better_mase": float(np.mean(out["mase_diff"] < 0)),
        "prob_chronosx_better_wql": float(np.mean(out["wql_diff"] < 0)),
        "win_rate_mase": float(np.mean(d_mase < 0)),  # % de series donde ChronosX tiene menor MASE
    }
