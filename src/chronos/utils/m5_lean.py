import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
from datasetsforecast.m5 import M5

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
COVARIATES = [
    "snap_CA", "snap_TX", "snap_WI", "sell_price",
    "event_type_1_Cultural", "event_type_1_National", "event_type_1_Religious",
    "event_type_1_Sporting", "event_type_1_nan",
    "event_type_2_Cultural", "event_type_2_Religious", "event_type_2_nan",
]
CHUNK = 500  # series por bloque escrito a disco


def _list_array(arrs, pa_type):
    """Lista de arrays numpy -> ListArray de Arrow sin pasar por objetos Python."""
    offsets = np.zeros(len(arrs) + 1, dtype=np.int32)
    np.cumsum([len(a) for a in arrs], out=offsets[1:])
    flat = pa.array(np.concatenate(arrs), type=pa_type)
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), flat)



def build_m5(num_series=0, seed=0, directory=".", out_dir=None):
    """num_series=0 -> las 30,490 series. Devuelve el Dataset (guardado en out_dir)."""
    M5.download(directory)
    path = f"{directory}/m5/datasets"
    if out_dir is None:
        tag = "full" if not num_series else f"n{num_series}_s{seed}"
        out_dir = Path(directory) / f"m5_lean_{tag}"
    out_dir = Path(out_dir)
    if out_dir.is_dir():
        return datasets.load_from_disk(out_dir)

    # ---- calendario -> matriz (dias x covariables de calendario)
    cal = pd.read_csv(
        f"{path}/calendar.csv",
        usecols=["date", "wm_yr_wk", "event_type_1", "event_type_2",
                 "snap_CA", "snap_TX", "snap_WI"],
        parse_dates=["date"],
    )
    for c in ["event_type_1", "event_type_2"]:
        cal[c] = cal[c].fillna("nan")
    dummies = pd.get_dummies(cal[["event_type_1", "event_type_2"]]).astype(np.float32)
    cal_feats = pd.concat(
        [cal[["snap_CA", "snap_TX", "snap_WI"]].astype(np.float32), dummies], axis=1
    )
    cal_names = [c for c in COVARIATES if c != "sell_price"]
    cal_mat = cal_feats[cal_names].to_numpy(np.float32)  # (ndays, 11)
    dates = cal["date"].to_numpy().astype("datetime64[ms]")
    ndays = len(cal)
    weeks = np.sort(cal["wm_yr_wk"].unique())
    day_week = np.searchsorted(weeks, cal["wm_yr_wk"].to_numpy())

    # ---- precios -> tensor (item, store, semana)
    prices = pd.read_csv(
        f"{path}/sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category",
               "wm_yr_wk": np.uint16, "sell_price": np.float32},
    )
    item_cats = prices["item_id"].cat.categories
    store_cats = prices["store_id"].cat.categories
    P = np.full((len(item_cats), len(store_cats), len(weeks)), np.nan, np.float32)
    P[
        prices["item_id"].cat.codes.to_numpy(),
        prices["store_id"].cat.codes.to_numpy(),
        np.searchsorted(weeks, prices["wm_yr_wk"].to_numpy()),
    ] = prices["sell_price"].to_numpy()
    del prices

    # ---- ventas (train + test evaluation)
    d_dtypes = {f"d_{i + 1}": np.float32 for i in range(ndays)}
    train = pd.read_csv(f"{path}/sales_train_evaluation.csv", dtype=d_dtypes)
    test = pd.read_csv(f"{path}/sales_test_evaluation.csv", dtype=d_dtypes)
    sales = train.merge(test, how="left", on=ID_COLS)
    del train, test
    sales["id"] = sales["item_id"].astype(str) + "_" + sales["store_id"].astype(str)
    sales = sales.sort_values("id").reset_index(drop=True)
    if num_series:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(sales), size=num_series, replace=False))
        sales = sales.iloc[sel].reset_index(drop=True)

    n = len(sales)
    d_cols = [f"d_{i + 1}" for i in range(ndays)]
    Y = sales[d_cols].to_numpy(np.float32)
    ids = sales["id"].tolist()
    static = {c: sales[c].astype(str).tolist() for c in ID_COLS}
    item_c = item_cats.get_indexer(sales["item_id"])
    store_c = store_cats.get_indexer(sales["store_id"])
    del sales
    PX = P[item_c[:, None], store_c[:, None], day_week[None, :]]  # (n, ndays)

    feats = datasets.Features({
        "id": datasets.Value("string"),
        "timestamp": datasets.Sequence(datasets.Value("timestamp[ms]")),
        "target": datasets.Sequence(datasets.Value("float32")),
        **{c: datasets.Sequence(datasets.Value("float32")) for c in COVARIATES},
        **{c: datasets.Value("string") for c in ID_COLS},
    })

    tmp = Path(str(out_dir) + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    parts = []
    for b, lo in enumerate(range(0, n, CHUNK)):
        hi = min(lo + CHUNK, n)
        rows = {k: [] for k in feats}
        for i in range(lo, hi):
            keep = np.maximum.accumulate(Y[i] > 0) & ~np.isnan(PX[i])
            rows["id"].append(ids[i])
            rows["timestamp"].append(dates[keep])
            rows["target"].append(Y[i][keep])
            for j, name in enumerate(cal_names):
                rows[name].append(cal_mat[keep, j])
            rows["sell_price"].append(PX[i][keep])
            for c in ID_COLS:
                rows[c].append(static[c][i])
        cols = {
            "id": pa.array(rows["id"], type=pa.string()),
            "timestamp": _list_array(rows["timestamp"], pa.timestamp("ms")),
            "target": _list_array(rows["target"], pa.float32()),
        }
        for c in COVARIATES:
            cols[c] = _list_array(rows[c], pa.float32())
        for c in ID_COLS:
            cols[c] = pa.array(rows[c], type=pa.string())
        part = datasets.Dataset(
            pa.table(cols), info=datasets.DatasetInfo(features=feats), split="train"
        )
        part.save_to_disk(tmp / f"part{b:04d}")
        parts.append(tmp / f"part{b:04d}")
        del rows, part
        print(f"  procesadas {hi}/{n} series", flush=True)

    ds = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in parts])
    ds.save_to_disk(out_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    return datasets.load_from_disk(out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_series", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = build_m5(a.num_series, a.seed)
    print(d)