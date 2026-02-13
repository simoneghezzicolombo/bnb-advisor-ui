import argparse
import gzip
import json
import math
import os
from datetime import datetime

import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_any_csv(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


def clean_price_to_float(s):
    # e.g. "€81.00" or "$120.00" or "81"
    if pd.isna(s):
        return None
    x = str(s).strip()
    x = x.replace("€", "").replace("$", "").replace("£", "").replace(",", "")
    try:
        return float(x)
    except:
        return None


def build_anchor_from_listings(df, center_lat, center_lon, radius_km,
                              accommodates_min, accommodates_max,
                              bedrooms_min, bedrooms_max,
                              room_type="Entire home/apt",
                              min_nights_max=2):
    # Basic cleanup / selection
    for col in ["latitude", "longitude", "price"]:
        if col not in df.columns:
            raise ValueError(f"Missing column in listings: {col}")

    # distance filter
    dists = []
    for lat, lon in zip(df["latitude"], df["longitude"]):
        try:
            dists.append(haversine_km(center_lat, center_lon, float(lat), float(lon)))
        except:
            dists.append(10**9)
    df = df.copy()
    df["dist_km"] = dists
    df = df[df["dist_km"] <= radius_km]

    # room type
    if "room_type" in df.columns:
        df = df[df["room_type"] == room_type]

    # accommodates
    if "accommodates" in df.columns:
        df = df[(df["accommodates"] >= accommodates_min) & (df["accommodates"] <= accommodates_max)]

    # bedrooms (can be NaN)
    if "bedrooms" in df.columns:
        df = df[(df["bedrooms"].fillna(-1) >= bedrooms_min) & (df["bedrooms"].fillna(-1) <= bedrooms_max)]

    # minimum nights
    if "minimum_nights" in df.columns:
        df = df[df["minimum_nights"].fillna(999) <= min_nights_max]

    # price
    df["price_num"] = df["price"].apply(clean_price_to_float)
    df = df[df["price_num"].notna() & (df["price_num"] > 0)]

    if len(df) < 30:
        # not enough comps, but still return something
        # (caller can widen radius or relax filters)
        pass

    # Robust stats
    p50 = float(df["price_num"].median()) if len(df) else None
    p25 = float(df["price_num"].quantile(0.25)) if len(df) else None
    p75 = float(df["price_num"].quantile(0.75)) if len(df) else None

    # Convert "full apt price" to a 2-guest anchor:
    # assume part is fixed, part scales with guests
    # (simple, stable for early stage)
    # fixed_share=0.55 means 55% fixed, 45% scales with guests
    fixed_share = 0.55
    assumed_capacity = 7  # your max
    if p50 is not None:
        per_guest = p50 / assumed_capacity
        anchor_2p = (p50 * fixed_share) + (per_guest * 2 * (1 - fixed_share))
        anchor_2p = round(anchor_2p)
    else:
        anchor_2p = None

    return {
        "comps_count": int(len(df)),
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "anchor_2p_weekday_guess": anchor_2p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listings", required=True, help="Path to listings.csv or listings.csv.gz")
    ap.add_argument("--center-lat", type=float, required=True)
    ap.add_argument("--center-lon", type=float, required=True)
    ap.add_argument("--radius-km", type=float, default=40.0)

    ap.add_argument("--acc-min", type=int, default=6)
    ap.add_argument("--acc-max", type=int, default=8)
    ap.add_argument("--bed-min", type=int, default=2)
    ap.add_argument("--bed-max", type=int, default=3)

    ap.add_argument("--out", default="market_anchor.json")
    args = ap.parse_args()

    df = load_any_csv(args.listings)

    anchor = build_anchor_from_listings(
        df,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        radius_km=args.radius_km,
        accommodates_min=args.acc_min,
        accommodates_max=args.acc_max,
        bedrooms_min=args.bed_min,
        bedrooms_max=args.bed_max,
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "center": {"lat": args.center_lat, "lon": args.center_lon},
        "radius_km": args.radius_km,
        "filters": {
            "accommodates": [args.acc_min, args.acc_max],
            "bedrooms": [args.bed_min, args.bed_max],
            "room_type": "Entire home/apt",
            "minimum_nights_max": 2,
        },
        "anchor": anchor,
        "notes": [
            "anchor_2p_weekday_guess is a conservative proxy derived from listing median price.",
            "You can widen radius-km or relax filters if comps_count is low."
        ],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OK → scritto {args.out}")
    print(f"Comps: {payload['anchor']['comps_count']}, p50={payload['anchor']['p50']}, anchor_2p={payload['anchor']['anchor_2p_weekday_guess']}")


if __name__ == "__main__":
    main()
