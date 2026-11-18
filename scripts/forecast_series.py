"""
Pronosticos de unas pocas series con ChronosX ya entrenado (para graficar).
No entrena nada: usa <cx_root>/final/run=<run>/final-checkpoint.

Si no se pasan --ids, elige por categoria (FOODS, HOBBIES, HOUSEHOLD) una serie de alto volumen y una
intermitente. Guarda <out> (npz) con: ids, inicio del pronostico, historia, valor real del test y los
cuantiles 0.1..0.9 de ChronosX. El notebook de UniCA lee ese archivo para pronosticar las MISMAS series
con UniCA y despues grafica los dos modelos.

Uso (desde la raiz del repo):
    python scripts/forecast_series.py --cx_root /content/drive/MyDrive/chronosx/m5 \
        --out /content/drive/MyDrive/chronosx/m5/forecasts/cx_forecasts.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from gluonts.dataset.split import split

from cx_utils import LazyUnivariate, _get_hf_dataset, load_cfg

p = argparse.ArgumentParser()
p.add_argument("--cx_root", required=True)
p.add_argument("--run", default="run=0")
p.add_argument("--ids", nargs="*", default=None, help="ids de M5 (item_store), p.ej. FOODS_3_090_CA_3")
p.add_argument("--history", type=int, default=120, help="dias de historia que se guardan para el grafico")
p.add_argument("--injection", default="IIB+OIB")
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()

Q9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cfg = load_cfg("m5_with_covariates")
H = cfg["prediction_length"]

print("Cargando M5 (la primera vez en la sesion lo descarga y lo arma, ~10-20 min)...", flush=True)
hf = _get_hf_dataset(cfg, 0, 0)
all_ids = np.array(hf.data.column("id").to_pylist())

# ---------------------------------------------------------------- series a graficar
if a.ids:
    ids = list(a.ids)
    missing = [s for s in ids if s not in set(all_ids)]
    assert not missing, f"no existen en M5: {missing}"
else:
    rng = np.random.default_rng(a.seed)
    mean365, zeros365 = np.zeros(len(hf)), np.zeros(len(hf))
    i = 0
    for b in hf.select_columns(["target"]).with_format("numpy").iter(batch_size=1000):
        for y in b["target"]:
            last = np.asarray(y, float)[:-H][-365:]
            mean365[i] = last.mean() if len(last) else 0
            zeros365[i] = (last == 0).mean() if len(last) else 1
            i += 1
    cat = np.array([s.split("_")[0] for s in all_ids])
    ids = []
    for c in ("FOODS", "HOBBIES", "HOUSEHOLD"):
        sel = np.where(cat == c)[0]
        ids.append(all_ids[sel[np.argmax(mean365[sel])]])  # alto volumen
        inter = sel[(zeros365[sel] >= 0.5) & (zeros365[sel] < 0.8) & (mean365[sel] >= 0.4)]
        if len(inter):
            ids.append(all_ids[rng.choice(inter)])  # intermitente
print("Series:", ids, flush=True)

pos = {s: i for i, s in enumerate(all_ids)}
sub = hf.select([pos[s] for s in ids])
lazy = LazyUnivariate(sub, cfg["series_fields"], cfg.get("covariates_fields"))
_, tmpl = split(lazy, offset=cfg["offset"])
test = tmpl.generate_instances(H, windows=1)
inputs, labels = list(test.input), list(test.label)

hist = np.full((len(ids), a.history), np.nan, np.float32)
for i, e in enumerate(inputs):
    h = np.asarray(e["target"], np.float32)[-a.history:]
    hist[i, a.history - len(h):] = h
y = np.stack([np.asarray(l["target"], np.float32) for l in labels])
fstart = [str((e["start"] + len(e["target"])).to_timestamp().date()) for e in inputs]


from chronosx.chronosx import ChronosXPipeline  # noqa: E402


def quantiles(forecasts):
    return np.stack([np.stack([f.quantile(q) for q in Q9]) for f in forecasts]).astype(np.float32)  # (n, 9, H)


# ---------------------------------------------------------------- ChronosX entrenado

ckpt = Path(a.cx_root) / "final" / a.run / "final-checkpoint"
print("ChronosX desde", ckpt, flush=True)
pipe = ChronosXPipeline(prediction_length=H, num_covariates=2 * len(cfg.get("covariates_fields") or []),
                        covariate_injection=a.injection, pretrained_model_name_or_path=ckpt, device_map=a.device)
pipe.chronosx.eval()
torch.manual_seed(a.seed)
cx = quantiles(pipe.generate_forecasts(inputs, batch_size=16))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, ids=np.array(ids), fstart=np.array(fstart), hist=hist, y=y, q=np.array(Q9),
         **{"ChronosX": cx})
print("Guardado en", a.out)
