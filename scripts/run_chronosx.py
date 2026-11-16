"""Una corrida: Chronos zero-shot (baseline) vs ChronosX en M5. Normalmente se lanza desde orchestrate.py.

Ejecutar desde la raiz del repo:  python scripts/run_chronosx.py --num_series 300 --max_steps 20 --out ./output/prueba
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from cx_utils import (
    aggregate,
    batched,
    enable_light_eval,
    enable_trainer_resume,
    item_metrics,
    load_data,
    paired_bootstrap,
    training_stream,
    validation_entries,
)

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="m5_with_covariates", choices=["m5_with_covariates"])
p.add_argument("--num_series", type=int, default=0, help="0 = todas las series del dataset")
p.add_argument("--num_rolls", type=int, default=None, help="ventanas de test (por defecto, la del yaml)")
p.add_argument("--model", default="amazon/chronos-t5-small")
p.add_argument("--injection", default="IIB+OIB")
p.add_argument("--device", default="cuda")
p.add_argument("--lr", type=float, default=0.001)
p.add_argument("--max_steps", type=int, default=5000)
p.add_argument("--run_id", type=int, default=0, help="semilla de entrenamiento = seed + run_id")
p.add_argument("--seed", type=int, default=0, help="semilla de datos / muestreo de evaluacion")
p.add_argument("--eval_steps", type=int, default=250, help="el repo usa 100")
p.add_argument("--val_series", type=int, default=1000, help="series (aleatorias) para validacion")
p.add_argument("--train_batch", type=int, default=32, help="como el repo")
p.add_argument("--grad_accum", type=int, default=2, help="como el repo")
p.add_argument("--batch_size", type=int, default=16, help="series por lote al predecir (x20 muestras)")
p.add_argument("--save_total_limit", type=int, default=3, help="mas copias = mas margen si el ultimo no termino de subir a Drive")
p.add_argument("--n_boot", type=int, default=1000)
p.add_argument("--skip_test_eval", action="store_true")
p.add_argument("--out", default="./output/run")
p.add_argument("--baseline_dir", default="./output/baseline")
args = p.parse_args()

t0 = time.time()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if (out / "results.json").exists():
    print(f"[skip] {out}/results.json ya existe")
    sys.exit(0)
if args.skip_test_eval and (out / "train_done.json").exists():
    print(f"[skip] {out}/train_done.json ya existe")
    sys.exit(0)

train_seed = args.seed + args.run_id
timing = {}


def tick(name, t_start):
    timing[name] = round((time.time() - t_start) / 60, 2)
    print(f"[tiempo] {name}: {timing[name]} min", flush=True)


# ------------------------------------------------------------------ datos
t = time.time()
print(f"Cargando {args.dataset} (num_series={args.num_series})...", flush=True)
data = load_data(args.dataset, args.num_series, args.seed, args.num_rolls)
H, num_cov = data["H"], data["num_covariates"]
print(f"Series: {data['n_series']} | ventanas: {data['num_rolls']} | horizonte: {H} | "
      f"covariables (con mascara): {num_cov}", flush=True)
tick("carga_datos", t)


# ------------------------------------------------------------------ 1. baseline
def chronos_zero_shot_forecasts():
    from chronos import ChronosPipeline
    from gluonts.model.forecast import SampleForecast

    pipe = ChronosPipeline.from_pretrained(args.model, device_map=args.device)
    torch.manual_seed(args.seed)
    forecasts = []
    for batch in batched(data["test_data"].input, args.batch_size):
        ctx = [torch.tensor(e["target"][-512:], dtype=torch.float32) for e in batch]
        samples = pipe.predict(ctx, prediction_length=H, num_samples=20).numpy()
        for e, s in zip(batch, samples):
            forecasts.append(SampleForecast(samples=s, start_date=e["start"] + len(e["target"])))
        if len(forecasts) % (50 * args.batch_size) == 0:
            print(f"  baseline: {len(forecasts)}/{data['n_series'] * data['num_rolls']}", flush=True)
    del pipe
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return forecasts


base_items = None
if not args.skip_test_eval:
    t = time.time()
    key = (f"{args.dataset}_n{args.num_series}_s{args.seed}_r{data['num_rolls']}_"
           f"{args.model.replace('/', '_')}")
    npz = Path(args.baseline_dir) / f"{key}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    print("\n[1/3] Baseline: Chronos zero-shot (sin covariables)", flush=True)
    if npz.exists():
        base_items = dict(np.load(npz))
        print(f"  (baseline cargado de {npz})", flush=True)
    else:
        base_fc = chronos_zero_shot_forecasts()
        base_items = item_metrics(base_fc, data["test_data"])
        np.savez(npz, **base_items)
        del base_fc
    tick("baseline", t)

# ------------------------------------------------------------------ 2. entrenamiento
from chronosx.chronosx import ChronosXPipeline  # noqa: E402
from chronosx.utils.chronos_dataset import ChronosDataset  # noqa: E402

print("\n[2/3] Entrenando adaptadores ChronosX", flush=True)
enable_light_eval()  # la validacion no acumula hidden states en RAM (ver cx_utils)
if (out / "train_done.json").exists():
    td = json.load(open(out / "train_done.json"))
    print("  (entrenamiento ya hecho)", flush=True)
else:
    t = time.time()
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    pipeline = ChronosXPipeline(
        prediction_length=H,
        num_covariates=num_cov,
        covariate_injection=args.injection,
        pretrained_model_name_or_path=args.model,
        device_map=args.device,
    )
    tok = pipeline.tokenizer
    val_entries = validation_entries(data, args.val_series, args.seed)
    print(f"  validacion: {len(val_entries)} series", flush=True)
    val_q = ChronosDataset(
        datasets=[val_entries], probabilities=[1.0], tokenizer=tok,
        prediction_length=H, min_past=1, mode="validation",
    )
    train_q = ChronosDataset(
        datasets=[training_stream(data)], probabilities=[1.0], tokenizer=tok,
        prediction_length=H, min_past=1, mode="training",
    ).shuffle(shuffle_buffer_length=100, random_seed=train_seed)

    zs_file = out / "zero_shot_val_loss.json"
    cached = json.load(open(zs_file)) if zs_file.exists() else {}
    if cached.get("n_val") == len(val_entries):  # solo reutiliza si se calculo con el mismo conjunto de validacion
        vl_pre = cached["val_loss"]
    else:
        vl_pre = float(pipeline.evaluate_model_on_validation_set(
            quantized_val_dataset=val_q, output_dir=out / "zs_eval",
            device_map=args.device, covariate_injection=None,
        ))
        json.dump({"val_loss": vl_pre, "n_val": len(val_entries)}, open(zs_file, "w"))
    print(f"  perdida de validacion zero-shot: {vl_pre:.4f}", flush=True)

    enable_trainer_resume()
    vl_ft = float(pipeline.train(
        output_dir=out,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        per_device_train_batch_size=args.train_batch,
        gradient_accumulation_steps=args.grad_accum,
        save_total_limit=args.save_total_limit,
        quantized_train_dataset=train_q,
        quantized_val_dataset=val_q,
    ))
    td = {
        "val_loss_zero_shot": vl_pre,
        "val_loss_chronosx": vl_ft,
        "adapters_beat_zero_shot_on_val": bool(vl_ft < vl_pre),
        "lr": args.lr,
        "max_steps": args.max_steps,
        "run_id": args.run_id,
        "train_seed": train_seed,
    }
    json.dump(td, open(out / "train_done.json", "w"), indent=1)
    for d in out.glob("checkpoint-*"):  # ahorra espacio en Drive; final-checkpoint se conserva
        shutil.rmtree(d, ignore_errors=True)
    del pipeline
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    tick("entrenamiento", t)

print(f"  val_loss zero-shot={td['val_loss_zero_shot']:.4f} | ChronosX={td['val_loss_chronosx']:.4f} | "
      f"adaptadores mejoran en validacion: {td['adapters_beat_zero_shot_on_val']}", flush=True)
if not td["adapters_beat_zero_shot_on_val"]:
    print("  AVISO: el repo descartaria estos adaptadores (finetune() guarda el zero-shot si gana en "
          "validacion). Aqui se evalua igualmente el modelo afinado.", flush=True)

if args.skip_test_eval:
    print("[fin] --skip_test_eval: solo entrenamiento.", flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ 3. evaluacion
t = time.time()
print("\n[3/3] Evaluando ChronosX", flush=True)
pipeline = ChronosXPipeline(
    prediction_length=H,
    num_covariates=num_cov,
    covariate_injection=args.injection,
    pretrained_model_name_or_path=out / "final-checkpoint",
    device_map=args.device,
)
pipeline.chronosx.eval()
torch.manual_seed(args.seed)
fc = pipeline.generate_forecasts(data["test_data"].input, batch_size=args.batch_size)
items = item_metrics(fc, data["test_data"])
np.savez(out / "item_metrics.npz", **items)
base_agg, cx_agg = aggregate(base_items), aggregate(items)
boot = paired_bootstrap(base_items, items, n_boot=args.n_boot, seed=args.seed)
tick("evaluacion", t)

rel = lambda a, b: 100 * (b - a) / a  # noqa: E731
res = {
    "dataset": args.dataset,
    "num_series": args.num_series,
    "n_series": data["n_series"],
    "n_items": base_agg["n_items"],
    "num_rolls": data["num_rolls"],
    "lr": args.lr,
    "max_steps": args.max_steps,
    "run_id": args.run_id,
    "seed": args.seed,
    "eval_steps": args.eval_steps,
    "val_series": args.val_series,
    "train_batch": args.train_batch,
    "grad_accum": args.grad_accum,
    "model": args.model,
    "injection": args.injection,
    **{k: td[k] for k in ("val_loss_zero_shot", "val_loss_chronosx", "adapters_beat_zero_shot_on_val")},
    "baseline": base_agg,
    "chronosx": cx_agg,
    "rel_change_pct": {
        "MASE": rel(base_agg["MASE[0.5]"], cx_agg["MASE[0.5]"]),
        "WQL": rel(base_agg["mean_weighted_sum_quantile_loss"], cx_agg["mean_weighted_sum_quantile_loss"]),
    },
    "paired_bootstrap": boot,
    "timing_min": {**timing, "total": round((time.time() - t0) / 60, 2)},
}
json.dump(res, open(out / "results.json", "w"), indent=1)

print("\n===== RESULTADOS (menor es mejor) =====")
print("Chronos zero-shot:", base_agg)
print("ChronosX         :", cx_agg)
print("Cambio relativo %:", res["rel_change_pct"])
print("Bootstrap emparejado:", boot if boot else "n/a (menos de 30 series)")
print("Resultados guardados en", out / "results.json")
