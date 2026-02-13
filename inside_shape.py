import argparse
import json
import pandas as pd


def price_to_float(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.replace("€", "").replace("$", "").replace("£", "")
    s = s.replace(",", "")  # 1,234.00
    try:
        return float(s)
    except:
        return None


def avail_to_bool(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in ("t", "true", "1", "yes", "y")


def load_listings(path):
    df = pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)
    # Normalize column names we need
    need = ["id", "room_type", "accommodates", "bedrooms"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"listings missing column: {c}")
    return df


def load_calendar(path):
    df = pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)
    # Expected columns in InsideAirbnb calendar: listing_id, date, available, price
    for c in ["listing_id", "date", "available", "price"]:
        if c not in df.columns:
            raise ValueError(f"calendar missing column: {c}")
    return df


def compute_shape_for_dataset(listings_path, calendar_path,
                              acc_min=6, acc_max=8, bed_min=2, bed_max=3,
                              room_type="Entire home/apt"):
    L = load_listings(listings_path).copy()

    # Filter comparable-ish inventory
    L = L[L["room_type"] == room_type]
    L = L[(L["accommodates"] >= acc_min) & (L["accommodates"] <= acc_max)]
    # bedrooms can be NaN
    L = L[(L["bedrooms"].fillna(-1) >= bed_min) & (L["bedrooms"].fillna(-1) <= bed_max)]

    keep_ids = set(L["id"].astype(int).tolist())
    if not keep_ids:
        return None

    C = load_calendar(calendar_path).copy()
    C = C[C["listing_id"].astype(int).isin(keep_ids)]

    # Parse
    C["date"] = pd.to_datetime(C["date"], errors="coerce")
    C = C[C["date"].notna()]

    C["available"] = C["available"].apply(avail_to_bool)
    C["price_num"] = C["price"].apply(price_to_float)
    C = C[C["price_num"].notna() & (C["price_num"] > 0)]

    # Use only available nights to estimate "posted price shape"
    C = C[C["available"]]

    if len(C) < 200:
        # Still can work, but warn (caller can enlarge filters)
        pass

    C["weekday"] = C["date"].dt.weekday  # Mon=0 .. Sun=6
    # Weekend nights: Fri (4), Sat (5)
    wknd = C[C["weekday"].isin([4, 5])]
    wkdy = C[~C["weekday"].isin([4, 5])]

    if len(wknd) < 20 or len(wkdy) < 50:
        weekend_uplift = None
    else:
        weekend_uplift = float(wknd["price_num"].median() / wkdy["price_num"].median())

    # Month multipliers: median weekday price in month vs overall weekday median
    base_med = float(wkdy["price_num"].median()) if len(wkdy) else None
    month_mult = {}
    if base_med and base_med > 0:
        for m in range(1, 13):
            mm = wkdy[wkdy["date"].dt.month == m]
            if len(mm) >= 20:
                month_mult[str(m)] = float(mm["price_num"].median() / base_med)
            else:
                month_mult[str(m)] = 1.0
    else:
        month_mult = {str(m): 1.0 for m in range(1, 13)}

    return {
        "n_calendar_rows_used": int(len(C)),
        "weekend_uplift": weekend_uplift,
        "month_mult": month_mult
    }


def weighted_avg(vals):
    # vals: list of (value, weight)
    vs = [(v, w) for v, w in vals if v is not None and w and w > 0]
    if not vs:
        return None
    num = sum(v * w for v, w in vs)
    den = sum(w for _, w in vs)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=2, action="append", required=True, metavar=("LISTINGS", "CALENDAR"))
    ap.add_argument("--out", default="inside_shape.json")

    ap.add_argument("--acc-min", type=int, default=6)
    ap.add_argument("--acc-max", type=int, default=8)
    ap.add_argument("--bed-min", type=int, default=2)
    ap.add_argument("--bed-max", type=int, default=3)

    args = ap.parse_args()

    shapes = []
    for listings_path, calendar_path in args.dataset:
        shape = compute_shape_for_dataset(
            listings_path, calendar_path,
            acc_min=args.acc_min, acc_max=args.acc_max,
            bed_min=args.bed_min, bed_max=args.bed_max
        )
        shapes.append({"listings": listings_path, "calendar": calendar_path, "shape": shape})

    # Combine: weights = number of calendar rows used
    wknd_vals = []
    month_vals = {str(m): [] for m in range(1, 13)}

    for item in shapes:
        sh = item["shape"]
        if not sh:
            continue
        w = sh["n_calendar_rows_used"]
        wknd_vals.append((sh.get("weekend_uplift"), w))
        for m in range(1, 13):
            month_vals[str(m)].append((sh["month_mult"].get(str(m), 1.0), w))

    weekend_uplift = weighted_avg(wknd_vals)
    month_mult = {str(m): float(weighted_avg(month_vals[str(m)]) or 1.0) for m in range(1, 13)}

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "filters": {
            "room_type": "Entire home/apt",
            "accommodates": [args.acc_min, args.acc_max],
            "bedrooms": [args.bed_min, args.bed_max],
            "weekend_days": ["Fri", "Sat"]
        },
        "combined": {
            "weekend_uplift": weekend_uplift,
            "month_mult": month_mult
        },
        "datasets": shapes
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"OK -> scritto {args.out}")
    print(f"Weekend uplift stimato: {weekend_uplift}")
    print("Month mult (1..12):", month_mult)


if __name__ == "__main__":
    main()
