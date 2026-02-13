"""
inside_shape_availability.py
----------------------------
Builds seasonality + weekend uplift for your BnB Advisor WITHOUT needing prices.

It uses InsideAirbnb "calendar.csv.gz" availability as a DEMAND proxy:
- month_unavail_rate (weekday) -> month_mult via elasticity
- weekend_unavail_rate vs weekday -> optional weekend_uplift (but we default to 1.25)

Output file is compatible with your GUI: inside_shape.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import pandas as pd


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_listings(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        compression="gzip" if path.endswith(".gz") else None,
        usecols=lambda c: c in {"id", "room_type", "accommodates", "bedrooms"},
    )
    df["id_num"] = pd.to_numeric(df["id"], errors="coerce")
    df = df[df["id_num"].notna()].copy()
    df["id_num"] = df["id_num"].astype("int64")
    return df


def load_calendar(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        compression="gzip" if path.endswith(".gz") else None,
        usecols=["listing_id", "date", "available"],
    )
    df["listing_id_num"] = pd.to_numeric(df["listing_id"], errors="coerce")
    df = df[df["listing_id_num"].notna()].copy()
    df["listing_id_num"] = df["listing_id_num"].astype("int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    # available is typically 't'/'f'
    df["avail_bool"] = df["available"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    df["unavail"] = ~df["avail_bool"]
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.weekday  # Mon=0 .. Sun=6
    return df


def shape_from_availability(
    listings_path: str,
    calendar_path: str,
    room_type: str,
    acc_min: int,
    acc_max: int,
    bed_min: int,
    bed_max: int,
) -> Optional[dict]:
    L = load_listings(listings_path)
    L = L[L["room_type"] == room_type]
    L = L[(L["accommodates"] >= acc_min) & (L["accommodates"] <= acc_max)]
    L = L[(L["bedrooms"].fillna(-1) >= bed_min) & (L["bedrooms"].fillna(-1) <= bed_max)]
    ids = set(L["id_num"].tolist())
    if not ids:
        return None

    C = load_calendar(calendar_path)
    C = C[C["listing_id_num"].isin(ids)].copy()
    if len(C) == 0:
        return None

    # Weekend as Fri+Sat nights
    wknd = C[C["dow"].isin([4, 5])]
    wkdy = C[~C["dow"].isin([4, 5])]

    ur_wknd = float(wknd["unavail"].mean()) if len(wknd) else None
    ur_wkdy = float(wkdy["unavail"].mean()) if len(wkdy) else None

    month_ur = {}
    for m in range(1, 13):
        mm = wkdy[wkdy["month"] == m]
        month_ur[str(m)] = float(mm["unavail"].mean()) if len(mm) else None

    return {
        "n_listings_kept": int(len(ids)),
        "n_calendar_rows_used": int(len(C)),
        "unavail_weekend": ur_wknd,
        "unavail_weekday": ur_wkdy,
        "month_unavail_weekday": month_ur,
    }


def to_price_shape(
    avail_shape: dict,
    elasticity: float,
    weekend_uplift_default: float,
    weekend_uplift_mode: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Map demand proxy -> price multipliers.
    price_mult = (unavail_ratio) ** elasticity
    """
    base = avail_shape.get("unavail_weekday", None) or None
    month_ur = avail_shape.get("month_unavail_weekday", {}) or {}

    month_mult: Dict[str, float] = {str(m): 1.0 for m in range(1, 13)}
    if base and base > 0:
        for m in range(1, 13):
            ur = month_ur.get(str(m), None)
            if ur is None:
                continue
            ratio = clamp(float(ur) / float(base), 0.55, 1.80)
            month_mult[str(m)] = float(ratio ** elasticity)

    if weekend_uplift_mode == "derived":
        ur_wknd = avail_shape.get("unavail_weekend", None)
        if base and base > 0 and ur_wknd is not None:
            ratio = clamp(float(ur_wknd) / float(base), 0.85, 1.50)
            weekend_uplift = float(ratio ** elasticity)
        else:
            weekend_uplift = float(weekend_uplift_default)
    else:
        weekend_uplift = float(weekend_uplift_default)

    return weekend_uplift, month_mult


def weighted_avg(pairs):
    pairs = [(v, w) for v, w in pairs if v is not None and w and w > 0]
    if not pairs:
        return None
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=2, action="append", required=True, metavar=("LISTINGS", "CALENDAR"))
    ap.add_argument("--out", default="inside_shape.json")

    ap.add_argument("--room-type", default="Entire home/apt")
    ap.add_argument("--acc-min", type=int, default=6)
    ap.add_argument("--acc-max", type=int, default=8)
    ap.add_argument("--bed-min", type=int, default=2)
    ap.add_argument("--bed-max", type=int, default=3)

    ap.add_argument("--elasticity", type=float, default=0.7,
                    help="How strongly demand proxy affects price. 0.6–0.9 is typical.")
    ap.add_argument("--weekend-uplift-default", type=float, default=1.25,
                    help="Fallback weekend uplift (Fri+Sat).")
    ap.add_argument("--weekend-uplift-mode", choices=["default", "derived"], default="default",
                    help="Use default uplift or derive from availability.")
    args = ap.parse_args()

    shapes = []
    for lp, cp in args.dataset:
        sh = shape_from_availability(
            listings_path=lp,
            calendar_path=cp,
            room_type=args.room_type,
            acc_min=args.acc_min,
            acc_max=args.acc_max,
            bed_min=args.bed_min,
            bed_max=args.bed_max,
        )
        if sh is None:
            shapes.append({"listings": lp, "calendar": cp, "error": "No rows after filters"})
        else:
            shapes.append({"listings": lp, "calendar": cp, "avail_shape": sh})

    wknd_uplifts = []
    month_mult_pairs = {str(m): [] for m in range(1, 13)}
    debug = []

    for item in shapes:
        if "avail_shape" not in item:
            debug.append(item)
            continue
        a = item["avail_shape"]
        w = a["n_calendar_rows_used"]
        wknd_uplift, mm = to_price_shape(
            a,
            elasticity=args.elasticity,
            weekend_uplift_default=args.weekend_uplift_default,
            weekend_uplift_mode=args.weekend_uplift_mode,
        )
        wknd_uplifts.append((wknd_uplift, w))
        for m in range(1, 13):
            month_mult_pairs[str(m)].append((mm[str(m)], w))

        debug.append({
            "listings": os.path.basename(item["listings"]),
            "calendar": os.path.basename(item["calendar"]),
            "n_listings_kept": a["n_listings_kept"],
            "n_calendar_rows_used": a["n_calendar_rows_used"],
            "unavail_weekday": a["unavail_weekday"],
            "unavail_weekend": a["unavail_weekend"],
        })

    combined_weekend = weighted_avg(wknd_uplifts) or args.weekend_uplift_default
    combined_month = {str(m): float(weighted_avg(month_mult_pairs[str(m)]) or 1.0) for m in range(1, 13)}

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "source": "inside_airbnb_availability_proxy",
        "filters": {
            "room_type": args.room_type,
            "accommodates": [args.acc_min, args.acc_max],
            "bedrooms": [args.bed_min, args.bed_max],
            "weekend_days": ["Fri", "Sat"],
            "elasticity": args.elasticity,
            "weekend_uplift_mode": args.weekend_uplift_mode,
        },
        "combined": {
            "weekend_uplift": float(combined_weekend),
            "month_mult": combined_month
        },
        "debug": debug
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"OK -> scritto {args.out}")
    print("Weekend uplift:", round(float(combined_weekend), 3))
    print("Month mult (1..12):", {m: round(float(combined_month[str(m)]), 2) for m in range(1, 13)})


if __name__ == "__main__":
    main()
