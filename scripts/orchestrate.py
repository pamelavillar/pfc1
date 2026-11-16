"""Barrido de lr + corridas finales + resumen de ChronosX en M5 (reanudable).

Ejecutar desde la raiz del repo:  python scripts/orchestrate.py --out_root ./output/m5 [--num_series 300 ...]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="m5_with_covariates", choices=["m5_with_covariates"])
p.add_argument("--num_series", type=int, default=0)
p.add_argument("--num_rolls", type=int, default=None)
p.add_argument("--out_root", required=True)
p.add_argument("--sweep_lrs", type=float, nargs="*", default=[0.01, 0.001])
p.add_argument("--sweep_steps", type=int, default=1500)
p.add_argument("--best_lr", type=float, default=None, help="si se da, se salta el barrido")
p.add_argument("--final_steps", type=int, default=5000)
p.add_argument("--final_runs", type=int, default=3)
p.add_argument("--model", default="amazon/chronos-t5-small")
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval_steps", type=int, default=250)
p.add_argument("--val_series", type=int, default=1000)
p.add_argument("--train_batch", type=int, default=32)
p.add_argument("--grad_accum", type=int, default=2)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--n_boot", type=int, default=1000)
p.add_argument("--summarize_only", action="store_true")
args = p.parse_args()

root = Path(args.out_root)
(root / "logs").mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent

PAPER_M5 = {  # valores que dio el usuario (no verificados contra el paper)
    "Chronos": {"WQL": 0.549, "MASE": 0.919},
    "ChronosX": {"WQL": 0.557, "MASE": 0.925},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(label, out_dir, extra):
    cmd = [
        sys.executable, "-u", str(here / "run_chronosx.py"),
        "--dataset", args.dataset, "--num_series", str(args.num_series),
        "--model", args.model, "--device", args.device, "--seed", str(args.seed),
        "--eval_steps", str(args.eval_steps), "--val_series", str(args.val_series),
        "--train_batch", str(args.train_batch), "--grad_accum", str(args.grad_accum),
        "--batch_size", str(args.batch_size), "--n_boot", str(args.n_boot),
        "--out", str(out_dir), "--baseline_dir", str(root / "baseline"),
    ] + (["--num_rolls", str(args.num_rolls)] if args.num_rolls else []) + extra
    logfile = root / "logs" / f"{label}.log"
    log(f"{label}: lanzando (log: {logfile})")
    t = time.time()
    with open(logfile, "a") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    log(f"{label}: termino con codigo {rc} en {(time.time() - t) / 60:.1f} min")
    if rc != 0:
        log(f"ERROR en {label}. Revisa el final de {logfile}. Vuelve a lanzar el mismo comando "
            f"para reanudar (retoma desde el ultimo checkpoint).")
        sys.exit(rc)


def summarize():
    runs = []
    for d in sorted((root / "final").glob("run=*")):
        f = d / "results.json"
        if f.exists():
            runs.append(json.load(open(f)))
    if not runs:
        log("No hay corridas finales terminadas todavia.")
        return
    state = json.load(open(root / "state.json")) if (root / "state.json").exists() else {}
    b = runs[0]["baseline"]
    L = []
    L.append(f"# Resumen: {args.dataset}  ({runs[0].get('n_series', '?')} series x {runs[0]['num_rolls']} ventana(s) "
             f"= {runs[0]['n_items']} pronosticos)\n")
    L.append(f"Pasos por corrida: {runs[0]['max_steps']} | lr elegido: {runs[0]['lr']} | "
             f"corridas terminadas: {len(runs)}/{args.final_runs}\n")
    if state.get("sweep"):
        L.append("Barrido de lr (perdida de validacion de ChronosX, menor es mejor): "
                 + ", ".join(f"lr={k}: {v:.4f}" for k, v in state["sweep"].items()) + "\n")
    L.append("| run | val_loss zero-shot | val_loss ChronosX | MASE Chronos | MASE ChronosX | cambio MASE | "
             "IC95 MASE | WQL Chronos | WQL ChronosX | cambio WQL | IC95 WQL |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    mase_cx, wql_cx = [], []
    for r in runs:
        cx, bo = r["chronosx"], r["paired_bootstrap"]
        mase_cx.append(cx["MASE[0.5]"])
        wql_cx.append(cx["mean_weighted_sum_quantile_loss"])
        ci_m = f"[{bo['mase_rel_pct_ci95'][0]:+.2f}, {bo['mase_rel_pct_ci95'][1]:+.2f}]%" if bo else "n/a"
        ci_w = f"[{bo['wql_rel_pct_ci95'][0]:+.2f}, {bo['wql_rel_pct_ci95'][1]:+.2f}]%" if bo else "n/a"
        L.append(
            f"| {r['run_id']} | {r['val_loss_zero_shot']:.4f} | {r['val_loss_chronosx']:.4f} | "
            f"{b['MASE[0.5]']:.4f} | {cx['MASE[0.5]']:.4f} | {r['rel_change_pct']['MASE']:+.2f}% | {ci_m} | "
            f"{b['mean_weighted_sum_quantile_loss']:.4f} | {cx['mean_weighted_sum_quantile_loss']:.4f} | "
            f"{r['rel_change_pct']['WQL']:+.2f}% | {ci_w} |"
        )
    sd = (lambda x: f"{np.std(x, ddof=1):.4f}" if len(x) > 1 else "n/a (1 sola corrida)")
    mm, mw = float(np.mean(mase_cx)), float(np.mean(wql_cx))
    L.append("")
    L.append(f"**Promedio de {len(runs)} corridas** (ChronosX): MASE {mm:.4f} +- {sd(mase_cx)} | "
             f"WQL {mw:.4f} +- {sd(wql_cx)}")
    L.append(f"**Chronos zero-shot**: MASE {b['MASE[0.5]']:.4f} | WQL {b['mean_weighted_sum_quantile_loss']:.4f}")
    L.append(f"**Cambio relativo del promedio**: MASE {100 * (mm - b['MASE[0.5]']) / b['MASE[0.5]']:+.2f}% | "
             f"WQL {100 * (mw - b['mean_weighted_sum_quantile_loss']) / b['mean_weighted_sum_quantile_loss']:+.2f}%")
    L.append("")
    L.append("IC95 = bootstrap emparejado (remuestreo de series; de ventanas si el dataset es de una sola serie) del "
             "cambio relativo de ChronosX frente a Chronos en esa corrida; si incluye 0, la diferencia no se "
             "distingue de ruido de muestreo. "
             "No incluye la variabilidad entre semillas de entrenamiento (para eso, mira la desviacion entre corridas).")
    if args.dataset == "m5_with_covariates":
        L.append("")
        L.append(f"Referencia (valores que me pasaste del paper, no verificados): Chronos WQL "
                 f"{PAPER_M5['Chronos']['WQL']} / MASE {PAPER_M5['Chronos']['MASE']}; ChronosX WQL "
                 f"{PAPER_M5['ChronosX']['WQL']} / MASE {PAPER_M5['ChronosX']['MASE']}.")
    tmin = [r["timing_min"].get("total") for r in runs]
    L.append("")
    L.append(f"Tiempo por corrida (min): {tmin}")
    (root / "summary.md").write_text("\n".join(L))
    json.dump({"runs": runs}, open(root / "summary.json", "w"), indent=1)
    print("\n".join(L))


if args.summarize_only:
    summarize()
    sys.exit(0)

state_f = root / "state.json"
state = json.load(open(state_f)) if state_f.exists() else {}

# ---------------------------------------------------------------- A. barrido
if args.best_lr is not None:
    best_lr = args.best_lr
    log(f"lr dado por el usuario: {best_lr} (sin barrido)")
else:
    if "best_lr" in state:
        best_lr = state["best_lr"]
        log(f"barrido ya hecho: mejor lr = {best_lr}  {state['sweep']}")
    else:
        sweep = {}
        for lr in args.sweep_lrs:
            d = root / "sweep" / f"lr={lr}"
            run(f"sweep_lr{lr}", d, ["--lr", str(lr), "--max_steps", str(args.sweep_steps),
                                      "--run_id", "0", "--skip_test_eval"])
            sweep[str(lr)] = json.load(open(d / "train_done.json"))["val_loss_chronosx"]
        best_lr = float(min(sweep, key=sweep.get))
        state.update({"sweep": sweep, "best_lr": best_lr, "sweep_steps": args.sweep_steps})
        json.dump(state, open(state_f, "w"), indent=1)
        log(f"barrido: {sweep} -> mejor lr = {best_lr}")

# ---------------------------------------------------------------- B. corridas finales
for r in range(args.final_runs):
    d = root / "final" / f"run={r}"
    run(f"final_run{r}", d, ["--lr", str(best_lr), "--max_steps", str(args.final_steps), "--run_id", str(r)])

# ---------------------------------------------------------------- C. resumen
summarize()
