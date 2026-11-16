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
    b = runs[0]["baseline"]
    metrics = [("MASE", "MASE[0.5]"), ("WQL", "mean_weighted_sum_quantile_loss")]
    cx = {k: np.mean([r["chronosx"][k] for r in runs]) for _, k in metrics}

    L = [f"{args.dataset} ({runs[0].get('n_series', '?')} series)", ""]
    L.append("ChronosX:")
    L += [f"- {name}: {cx[k]:.4f}" for name, k in metrics]
    L.append("")
    L.append("Chronos:")
    L += [f"- {name}: {b[k]:.4f}" for name, k in metrics]
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
