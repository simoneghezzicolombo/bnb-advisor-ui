from __future__ import annotations

import csv
import json
import os
import time
# tkinter non è disponibile su Streamlit Cloud: rendiamo il modulo importabile comunque
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    import types
    tk = types.SimpleNamespace(Tk=object)  # serve solo per class App(tk.Tk)
    ttk = types.SimpleNamespace()
    messagebox = types.SimpleNamespace()
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple, Dict

import requests

# ==========================
# FILES
# ==========================
CONFIG_FILE = "gui_config.json"
STATE_FILE = "advisor_state.json"
MARKET_PRIORS_FILE = "market_priors.json"
INSIDE_SHAPE_FILE = "inside_shape.json"
PEAK_WEEKS_FILE = "peak_weeks.json"

# ==========================
# DEFAULT RULES (fallbacks)
# ==========================
DIRECT_DISCOUNT = 0.15          # Sito sempre -15% vs OTA effective
DEFAULT_WEEKEND_UPLIFT = 1.25   # Weekend +25% (ven/sab)
PRICE_STEP_EUR = 5.0
MAX_STEP_PER_RUN = 2            # max 10€ per run

LAUNCH_FLOOR_EUR_DEFAULT = 60.0
LAUNCH_DAYS_DEFAULT = 21
LAUNCH_BOOKINGS_THRESHOLD_DEFAULT = 3

DEFAULT_MONTH_MULT = {
    1: 0.95, 2: 0.95, 3: 1.00, 4: 1.05, 5: 1.10, 6: 1.15,
    7: 1.35, 8: 1.35,
    9: 1.10, 10: 1.00, 11: 0.95, 12: 1.10
}

# Guest steps (2 -> 7 = +65€)
GUEST_STEP_3_5 = 15
GUEST_STEP_6_7 = 10

def delta_2p_to_7p_default() -> int:
    return 3 * GUEST_STEP_3_5 + 2 * GUEST_STEP_6_7

# ==========================
# HELPERS
# ==========================
def parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()

def parse_date_any(s: str) -> Optional[date]:
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None

def round_eur(x: float) -> int:
    return int(round(x))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def is_cancelled(status: str) -> bool:
    st = (status or "").strip().lower()
    return any(k in st for k in ["cancell", "annull", "void", "refunded", "rimbors"])

def fetch_csv_rows(url: str) -> list[dict]:
    sep = "&" if "?" in url else "?"
    url2 = f"{url}{sep}_cb={int(time.time())}"
    r = requests.get(url2, timeout=25)
    r.raise_for_status()
    reader = csv.DictReader(r.text.splitlines())
    return list(reader)

def find_key(keys: List[str], candidates: List[str]) -> Optional[str]:
    keys_l = [k.lower() for k in keys]
    for cand in candidates:
        c = cand.lower()
        for k in keys_l:
            if c == k:
                return k
        for k in keys_l:
            if c in k:
                return k
    return None

# -------- shape loaders
def load_inside_shape_combined() -> Optional[dict]:
    if not os.path.exists(INSIDE_SHAPE_FILE):
        return None
    try:
        with open(INSIDE_SHAPE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("combined")
    except Exception:
        return None

def compute_shape_overrides() -> Tuple[float, Dict[int, float], str]:
    shape = load_inside_shape_combined()
    if not shape:
        return DEFAULT_WEEKEND_UPLIFT, dict(DEFAULT_MONTH_MULT), "default"

    we = shape.get("weekend_uplift", None)
    mm = shape.get("month_mult", None)

    valid_we = isinstance(we, (int, float)) and 1.01 <= float(we) <= 2.0

    month_mult: Dict[int, float] = {}
    non_flat = False
    if isinstance(mm, dict) and len(mm) == 12:
        for k, v in mm.items():
            try:
                m = int(k)
                month_mult[m] = float(v)
            except Exception:
                pass
        if len(month_mult) == 12:
            non_flat = any(abs(month_mult[m] - 1.0) > 0.02 for m in range(1, 13))

    if not valid_we or not non_flat:
        return DEFAULT_WEEKEND_UPLIFT, dict(DEFAULT_MONTH_MULT), "fallback (inside_shape not informative)"

    return float(we), month_mult, "inside_shape.json"

# -------- peak weeks
def load_peak_weeks() -> dict:
    """
    Format:
    {
      "rules":[{"label":"Summer","start":"2026-07-01","end":"2026-08-31","mult":1.15}],
      "date_overrides":[{"date":"2026-12-31","mult":1.30,"label":"NYE"}]
    }
    """
    if not os.path.exists(PEAK_WEEKS_FILE):
        return {"rules": [], "date_overrides": []}
    try:
        with open(PEAK_WEEKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("rules", [])
        data.setdefault("date_overrides", [])
        return data
    except Exception:
        return {"rules": [], "date_overrides": []}

def peak_multiplier_for_day(d: date, peaks: dict) -> Tuple[float, str]:
    best_mult = 1.0
    best_label = ""
    # date overrides win
    for o in peaks.get("date_overrides", []):
        try:
            if parse_yyyy_mm_dd(o.get("date","")) == d:
                m = float(o.get("mult", 1.0))
                if m > best_mult:
                    best_mult = m
                    best_label = o.get("label","override")
        except Exception:
            continue
    for r in peaks.get("rules", []):
        try:
            s = parse_yyyy_mm_dd(r.get("start",""))
            e = parse_yyyy_mm_dd(r.get("end",""))
            if s <= d <= e:
                m = float(r.get("mult", 1.0))
                if m > best_mult:
                    best_mult = m
                    best_label = r.get("label","peak")
        except Exception:
            continue
    return best_mult, best_label

# -------- sheet schema
@dataclass
class SheetSchema:
    checkin: str
    checkout: str
    status: str

def infer_schema(rows: List[dict]) -> SheetSchema:
    if not rows:
        raise ValueError("CSV vuoto: controlla che il tab pubblicato abbia righe.")
    keys = [k.strip() for k in rows[0].keys()]
    checkin = find_key(keys, ["check-in", "check in", "checkin"])
    checkout = find_key(keys, ["check-out", "check out", "checkout"])
    status = find_key(keys, ["status", "stato"])
    if not checkin or not checkout or not status:
        raise ValueError("Servono colonne tipo: Check-in, Check-out, Status (anche nomi simili vanno bene).")
    return SheetSchema(checkin=checkin, checkout=checkout, status=status)

def build_blocked_nights(rows: List[dict], schema: SheetSchema) -> set[date]:
    blocked: set[date] = set()
    for row in rows:
        if is_cancelled(row.get(schema.status, "")):
            continue
        ci = parse_date_any(row.get(schema.checkin, ""))
        co = parse_date_any(row.get(schema.checkout, ""))
        if not ci or not co or co <= ci:
            continue
        d = ci
        while d < co:
            blocked.add(d)
            d += timedelta(days=1)
    return blocked

def occ_stats(blocked: set[date], start_day: date, horizon_days: int) -> Tuple[int, int, float]:
    booked = 0
    total = 0
    for i in range(horizon_days):
        d = start_day + timedelta(days=i)
        total += 1
        if d in blocked:
            booked += 1
    return booked, total, (booked / total if total else 0.0)

def count_future_bookings(rows: List[dict], schema: SheetSchema, from_date: date) -> int:
    c = 0
    for row in rows:
        if is_cancelled(row.get(schema.status, "")):
            continue
        ci = parse_date_any(row.get(schema.checkin, ""))
        co = parse_date_any(row.get(schema.checkout, ""))
        if not ci or not co or co <= ci:
            continue
        if ci >= from_date:
            c += 1
    return c

# -------- persistence
def load_state_default_base(default_base: float) -> float:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            s = json.load(f)
            return float(s.get("direct_base_weekday_2p", default_base))
    except Exception:
        return default_base

def save_state_base(base: float) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"direct_base_weekday_2p": float(base)}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_gui_config() -> dict:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_gui_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# -------- market priors
def load_market_priors() -> Optional[dict]:
    if not os.path.exists(MARKET_PRIORS_FILE):
        return None
    try:
        with open(MARKET_PRIORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def compute_market_anchor_from_priors(priors: dict) -> Optional[dict]:
    markets = priors.get("markets", [])
    if not markets:
        return None
    delta = priors.get("assumptions", {}).get("delta_2p_to_7p", None)
    if delta is None:
        delta = delta_2p_to_7p_default()

    w_sum = 0.0
    adr7_sum = 0.0
    occ_sum = 0.0
    used = []
    for m in markets:
        try:
            w = float(m.get("w", 0))
            occ = float(m.get("occ", 0))
            adr7 = float(m.get("adr_7p", 0))
        except Exception:
            continue
        if w <= 0 or adr7 <= 0 or occ <= 0:
            continue
        w_sum += w
        adr7_sum += w * adr7
        occ_sum += w * occ
        used.append({"name": m.get("name", "?"), "w": w, "occ": occ, "adr_7p": adr7})

    if w_sum <= 0:
        return None

    anchor_adr_7p = adr7_sum / w_sum
    target_occ_30 = occ_sum / w_sum
    target_occ_14 = min(0.85, target_occ_30 + 0.05)
    anchor_base_2p = max(25.0, anchor_adr_7p - float(delta))

    return {
        "delta_2p_to_7p": float(delta),
        "anchor_adr_7p": float(anchor_adr_7p),
        "anchor_base_2p": float(anchor_base_2p),
        "target_occ_14": float(target_occ_14),
        "target_occ_30": float(target_occ_30),
        "used_markets": used
    }

# -------- recommendations
def recommend_direct_base(
    current_base: float,
    occ14: float,
    occ30: float,
    target14: float,
    target30: float,
    allow_adjust: bool,
    launch_floor: float,
    in_launch: bool
) -> Tuple[float, str]:
    if not allow_adjust:
        base = current_base
        if in_launch:
            base = max(base, launch_floor)
            return base, f"Auto-adjust OFF + Launch floor {round_eur(launch_floor)}€ → mantengo {round_eur(base)}€."
        return base, "Auto-adjust OFF → mantengo il tuo base."

    score = 0
    if occ14 < target14 - 0.10:
        score -= 2
    elif occ14 < target14 - 0.05:
        score -= 1
    elif occ14 > target14 + 0.10:
        score += 2
    elif occ14 > target14 + 0.05:
        score += 1

    if occ30 < target30 - 0.10:
        score -= 1
    elif occ30 > target30 + 0.10:
        score += 1

    score = max(-MAX_STEP_PER_RUN, min(MAX_STEP_PER_RUN, score))
    new_base = max(35.0, current_base + score * PRICE_STEP_EUR)
    if in_launch:
        new_base = max(new_base, launch_floor)

    if score == 0:
        why = "Pacing ok → mantieni."
    elif score < 0:
        why = f"Pacing sotto target → abbassa di {abs(score)*PRICE_STEP_EUR:.0f}€."
    else:
        why = f"Pacing sopra target → alza di {abs(score)*PRICE_STEP_EUR:.0f}€."

    if in_launch and new_base == launch_floor and current_base + score * PRICE_STEP_EUR < launch_floor:
        why += f" (clamp Launch floor {round_eur(launch_floor)}€)"

    return new_base, why

def guest_price_for_n(base_2p: float, guests: int) -> float:
    if guests <= 2:
        return base_2p
    x = base_2p
    for g in range(3, guests + 1):
        x += (GUEST_STEP_3_5 if g <= 5 else GUEST_STEP_6_7)
    return x

def build_month_matrix(
    direct_base: float,
    face20: bool,
    airbnb_has20: bool,
    booking_has20: bool,
    weekend_uplift: float,
    month_mult: Dict[int, float],
) -> Tuple[list[dict], list[str]]:
    todo = []
    if face20 and not airbnb_has20:
        todo.append("Airbnb: crea una promo -20% quando disponibile (dopo le prime 3 prenotazioni) per mantenere l'effetto sconto.")
    if face20 and not booking_has20:
        todo.append("Booking: attiva una promo/deal -20% per mantenere l'effetto sconto (senza stacking).")

    ab_factor = 0.8 if (face20 and airbnb_has20) else 1.0
    bk_factor = 0.8 if (face20 and booking_has20) else 1.0

    matrix = []
    for m in range(1, 13):
        mult = float(month_mult.get(m, 1.0))
        direct_weekday = direct_base * mult
        direct_weekend = direct_base * mult * weekend_uplift

        ota_eff_weekday = direct_weekday / (1 - DIRECT_DISCOUNT)
        ota_eff_weekend = direct_weekend / (1 - DIRECT_DISCOUNT)

        airbnb_base = ota_eff_weekday / ab_factor
        airbnb_weekend = ota_eff_weekend / ab_factor
        booking_std = ota_eff_weekday / bk_factor
        booking_weekend = ota_eff_weekend / bk_factor

        matrix.append({
            "month": m,
            "direct_weekday_2p": round_eur(direct_weekday),
            "direct_weekend_2p": round_eur(direct_weekend),
            "airbnb_base_weekday_2p": round_eur(airbnb_base),
            "airbnb_weekend_2p": round_eur(airbnb_weekend),
            "booking_std_weekday_2p": round_eur(booking_std),
            "booking_std_weekend_2p": round_eur(booking_weekend),
        })

    return matrix, todo

def build_daily_matrix(
    start_day: date,
    days: int,
    direct_base: float,
    face20: bool,
    airbnb_has20: bool,
    booking_has20: bool,
    weekend_uplift: float,
    month_mult: Dict[int, float],
    peaks: dict
) -> list[dict]:
    ab_factor = 0.8 if (face20 and airbnb_has20) else 1.0
    bk_factor = 0.8 if (face20 and booking_has20) else 1.0

    out = []
    for i in range(days):
        d = start_day + timedelta(days=i)
        mult = float(month_mult.get(d.month, 1.0))
        is_wknd = d.weekday() in (4, 5)  # Fri/Sat
        wk = weekend_uplift if is_wknd else 1.0

        peak_mult, peak_label = peak_multiplier_for_day(d, peaks)

        direct_2p = direct_base * mult * wk * peak_mult
        ota_eff_2p = direct_2p / (1 - DIRECT_DISCOUNT)

        ab_2p = ota_eff_2p / ab_factor
        bk_2p = ota_eff_2p / bk_factor

        direct_7p = guest_price_for_n(direct_2p, 7)
        ab_7p = guest_price_for_n(ab_2p, 7)
        bk_7p = guest_price_for_n(bk_2p, 7)

        out.append({
            "date": d.isoformat(),
            "weekday": d.strftime("%a"),
            "is_weekend": int(is_wknd),
            "peak_mult": round(peak_mult, 3),
            "peak_label": peak_label,
            "direct_2p": round_eur(direct_2p),
            "airbnb_2p": round_eur(ab_2p),
            "booking_2p": round_eur(bk_2p),
            "direct_7p": round_eur(direct_7p),
            "airbnb_7p": round_eur(ab_7p),
            "booking_7p": round_eur(bk_7p),
        })
    return out

def write_outputs(
    go_live: str,
    pacing_from: str,
    feedback: dict,
    config: dict,
    matrix: list[dict],
    daily: list[dict],
    todo: list[str],
    checklist_text: str
) -> None:
    with open("matrix_monthly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "month","direct_weekday_2p","direct_weekend_2p",
            "airbnb_base_weekday_2p","airbnb_weekend_2p",
            "booking_std_weekday_2p","booking_std_weekend_2p"
        ])
        w.writeheader()
        for row in matrix:
            w.writerow(row)

    with open("matrix_daily.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date","weekday","is_weekend","peak_mult","peak_label",
            "direct_2p","airbnb_2p","booking_2p",
            "direct_7p","airbnb_7p","booking_7p"
        ])
        w.writeheader()
        for row in daily:
            w.writerow(row)

    recap = {
        "go_live_date": go_live,
        "pacing_from": pacing_from,
        "feedback": feedback,
        "config_used": config,
        "todo": todo,
        "matrix_monthly_2p": matrix,
        "matrix_daily": daily[:120],
        "checklist": checklist_text
    }
    with open("settings_recap.json", "w", encoding="utf-8") as f:
        json.dump(recap, f, ensure_ascii=False, indent=2)

    with open("recommendations.txt", "w", encoding="utf-8") as f:
        f.write("=== RECOMMENDATIONS ===\n")
        f.write(f"GO-LIVE: {go_live}\n")
        f.write(f"Pacing calcolato da: {pacing_from}\n\n")
        f.write(f"Occ 14 giorni: {feedback['occ14_booked']}/{feedback['occ14_total']} = {feedback['occ14_pct']:.0%} (target {feedback['target14']:.0%})\n")
        f.write(f"Occ 30 giorni: {feedback['occ30_booked']}/{feedback['occ30_total']} = {feedback['occ30_pct']:.0%} (target {feedback['target30']:.0%})\n\n")
        if feedback.get("benchmark_base_2p") is not None:
            f.write(f"Benchmark locale (Airbtics) base 2p: ~{round_eur(feedback['benchmark_base_2p'])}€\n")
            f.write(f"Tu (Direct base 2p): {feedback['direct_base_input']}€  => delta vs benchmark: {feedback['delta_vs_benchmark_pct']:+.0%}\n\n")
        f.write(f"Direct base usato (2p feriale): {feedback['direct_base_suggested']}€ (tu hai inserito: {feedback['direct_base_input']}€)\n")
        f.write(f"Motivo: {feedback['why']}\n")
        if feedback.get("in_launch"):
            f.write(f"Launch mode: ON (floor {feedback['launch_floor']}€)\n")
        f.write("\nFile generati: matrix_monthly.csv, matrix_daily.csv, settings_recap.json\n\n")
        if todo:
            f.write("TODO (per mantenere -20 senza stacking):\n")
            for t in todo:
                f.write(f"- {t}\n")
            f.write("\n")
        f.write("CHECKLIST (cosa impostare):\n")
        f.write(checklist_text + "\n")

# ==========================
# GUI
# ==========================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BnB Advisor — GUI (CSV + Airbtics + Seasonality + Peaks)")
        self.geometry("1060x860")

        cfg = load_gui_config()

        self.csv_url = tk.StringVar(value=cfg.get("csv_url", ""))
        self.go_live = tk.StringVar(value=cfg.get("go_live", "2026-02-10"))

        state_base = load_state_default_base(60.0)
        self.direct_base = tk.StringVar(value=str(cfg.get("direct_base", state_base)))
        self.auto_adjust = tk.BooleanVar(value=cfg.get("auto_adjust", True))

        self.launch_floor = tk.StringVar(value=str(cfg.get("launch_floor", LAUNCH_FLOOR_EUR_DEFAULT)))
        self.launch_days = tk.StringVar(value=str(cfg.get("launch_days", LAUNCH_DAYS_DEFAULT)))
        self.launch_bookings = tk.StringVar(value=str(cfg.get("launch_bookings", LAUNCH_BOOKINGS_THRESHOLD_DEFAULT)))

        self.face20 = tk.BooleanVar(value=cfg.get("face20", True))
        self.airbnb_has20 = tk.BooleanVar(value=cfg.get("airbnb_has20", True))
        self.booking_has20 = tk.BooleanVar(value=cfg.get("booking_has20", True))

        self.airbnb_weekly = tk.StringVar(value=str(cfg.get("airbnb_weekly_pct", 5)))
        self.booking_nonref = tk.StringVar(value=str(cfg.get("booking_nonref_pct", 10)))
        self.booking_weekly = tk.StringVar(value=str(cfg.get("booking_weekly_pct", 15)))

        self.daily_horizon = tk.StringVar(value=str(cfg.get("daily_horizon", 90)))

        self.last_matrix: Optional[list[dict]] = None
        self.last_daily: Optional[list[dict]] = None
        self.last_go_live_month: Optional[int] = None
        self.last_checklist_text: str = ""

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        ttk.Label(frm, text="CSV URL (Google Sheets pubblicato come CSV):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.csv_url, width=135).grid(row=1, column=0, columnspan=9, sticky="we")

        ttk.Label(frm, text="Go-live (YYYY-MM-DD):").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.go_live, width=16).grid(row=2, column=1, sticky="w")

        ttk.Label(frm, text="Direct base 2 ospiti feriale €:").grid(row=2, column=2, sticky="w")
        ttk.Entry(frm, textvariable=self.direct_base, width=8).grid(row=2, column=3, sticky="w")

        ttk.Checkbutton(frm, text="Auto-adjust (pacing)", variable=self.auto_adjust).grid(row=2, column=4, sticky="w")

        ttk.Label(frm, text="Daily horizon (giorni):").grid(row=2, column=5, sticky="w")
        ttk.Spinbox(frm, from_=14, to=365, textvariable=self.daily_horizon, width=6).grid(row=2, column=6, sticky="w")

        ttk.Separator(frm).grid(row=3, column=0, columnspan=9, sticky="we", pady=8)

        ttk.Label(frm, text="Launch floor €:").grid(row=4, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.launch_floor, width=8).grid(row=4, column=1, sticky="w")

        ttk.Label(frm, text="Launch giorni:").grid(row=4, column=2, sticky="w")
        ttk.Entry(frm, textvariable=self.launch_days, width=6).grid(row=4, column=3, sticky="w")

        ttk.Label(frm, text="Launch prenotazioni:").grid(row=4, column=4, sticky="w")
        ttk.Entry(frm, textvariable=self.launch_bookings, width=6).grid(row=4, column=5, sticky="w")

        ttk.Separator(frm).grid(row=5, column=0, columnspan=9, sticky="we", pady=8)

        ttk.Checkbutton(frm, text="Face -20% (senza stacking)", variable=self.face20).grid(row=6, column=0, sticky="w")
        ttk.Checkbutton(frm, text="Airbnb: -20% attivo ora", variable=self.airbnb_has20).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Booking: -20% attivo ora", variable=self.booking_has20).grid(row=6, column=2, sticky="w")

        ttk.Label(frm, text="Airbnb weekly %:").grid(row=7, column=0, sticky="w")
        ttk.Spinbox(frm, from_=0, to=30, textvariable=self.airbnb_weekly, width=6).grid(row=7, column=1, sticky="w")

        ttk.Label(frm, text="Booking nonref %:").grid(row=7, column=2, sticky="w")
        ttk.Spinbox(frm, from_=0, to=30, textvariable=self.booking_nonref, width=6).grid(row=7, column=3, sticky="w")

        ttk.Label(frm, text="Booking weekly %:").grid(row=7, column=4, sticky="w")
        ttk.Spinbox(frm, from_=0, to=30, textvariable=self.booking_weekly, width=6).grid(row=7, column=5, sticky="w")

        btns = ttk.Frame(frm)
        btns.grid(row=8, column=0, columnspan=9, sticky="w", pady=10)
        ttk.Button(btns, text="Run advisor", command=self.run).pack(side="left", padx=6)
        ttk.Button(btns, text="Apri cartella output", command=self.open_folder).pack(side="left", padx=6)
        ttk.Button(btns, text="Copia GO-LIVE month", command=self.copy_go_live_month).pack(side="left", padx=6)
        ttk.Button(btns, text="Copia checklist", command=self.copy_checklist).pack(side="left", padx=6)

        ttk.Separator(frm).grid(row=9, column=0, columnspan=9, sticky="we", pady=8)

        self.out = tk.Text(frm, height=30, wrap="word")
        self.out.grid(row=10, column=0, columnspan=9, sticky="nsew")
        frm.rowconfigure(10, weight=1)
        frm.columnconfigure(0, weight=1)

        ttk.Label(
            frm,
            text="Files letti: market_priors.json (benchmark), inside_shape.json (stagionalità), peak_weeks.json (eventi). Output: matrix_daily.csv + matrix_monthly.csv."
        ).grid(row=11, column=0, columnspan=9, sticky="w")

    def _append(self, s: str = ""):
        self.out.insert("end", s + "\n")
        self.out.see("end")

    def open_folder(self):
        os.startfile(os.getcwd())

    def _row_for_month(self, month: int) -> Optional[dict]:
        if not self.last_matrix:
            return None
        return next((x for x in self.last_matrix if x["month"] == month), None)

    def copy_go_live_month(self):
        if not self.last_matrix or not self.last_go_live_month:
            messagebox.showinfo("Info", "Prima premi Run advisor.")
            return
        row = self._row_for_month(self.last_go_live_month)
        if not row:
            messagebox.showinfo("Info", "Non trovo il mese go-live nella matrice.")
            return
        txt = (
            f"Mese GO-LIVE {self.last_go_live_month} (2 ospiti)\n"
            f"SITO Direct: weekday {row['direct_weekday_2p']} | weekend {row['direct_weekend_2p']}\n"
            f"AIRBNB: base {row['airbnb_base_weekday_2p']} | weekend {row['airbnb_weekend_2p']}\n"
            f"BOOKING: std {row['booking_std_weekday_2p']} | weekend {row['booking_std_weekend_2p']}\n"
        )
        self.clipboard_clear()
        self.clipboard_append(txt)
        messagebox.showinfo("Copiato", "Numeri mese GO-LIVE copiati negli appunti.")

    def copy_checklist(self):
        if not self.last_checklist_text:
            self.last_checklist_text = self._build_checklist_text()
        self.clipboard_clear()
        self.clipboard_append(self.last_checklist_text)
        messagebox.showinfo("Copiato", "Checklist copiata negli appunti.")

    def _build_checklist_text(self) -> str:
        aw = int(round(float(self.airbnb_weekly.get() or "5")))
        bn = int(round(float(self.booking_nonref.get() or "10")))
        bw = int(round(float(self.booking_weekly.get() or "15")))
        face20 = self.face20.get()

        delta = delta_2p_to_7p_default()
        guest_steps = f"3–5: +{GUEST_STEP_3_5}€/ospite | 6–7: +{GUEST_STEP_6_7}€/ospite (2→7 = +{delta}€)"

        lines = []
        lines.append("SITO (Direct)")
        lines.append(f"- Regola: Direct sempre -{int(DIRECT_DISCOUNT*100)}% vs OTA effective")
        lines.append("- Usa matrix_daily.csv per applicare prezzi giorno-per-giorno (include weekend + peak weeks).")
        lines.append("")
        lines.append("AIRBNB")
        lines.append("- Smart Pricing: OFF")
        lines.append("- Bulk edit calendario: usa colonna 'airbnb_2p' (2 ospiti) e gli scatti extra guest")
        lines.append(f"- Weekly discount: {aw}%")
        lines.append(f"- Extra guests: {guest_steps}")
        lines.append(f"- Face -20%: {'ON' if face20 else 'OFF'} (NO stacking)")
        lines.append("")
        lines.append("BOOKING")
        lines.append("- Availability planner: usa colonna 'booking_2p' (weekday/weekend/peak già inclusi)")
        lines.append(f"- Non-refundable: {bn}% più economico dello Standard")
        lines.append(f"- Weekly: {bw}% più economico dello Standard")
        lines.append(f"- Pricing per ospite: {guest_steps}")
        return "\n".join(lines)

    def run(self):
        self.out.delete("1.0", "end")

        url = self.csv_url.get().strip()
        if not url.startswith("http"):
            messagebox.showerror("Errore", "Incolla un CSV URL valido (deve iniziare con http).")
            return

        try:
            go_live_dt = parse_yyyy_mm_dd(self.go_live.get().strip())
        except Exception:
            messagebox.showerror("Errore", "Go-live deve essere YYYY-MM-DD (es: 2026-02-10).")
            return

        try:
            base_input = float(self.direct_base.get().strip().replace(",", "."))
        except Exception:
            messagebox.showerror("Errore", "Direct base deve essere un numero (es: 60).")
            return

        try:
            launch_floor = float(self.launch_floor.get().strip().replace(",", "."))
            launch_days = int(float(self.launch_days.get().strip()))
            launch_bookings_th = int(float(self.launch_bookings.get().strip()))
        except Exception:
            messagebox.showerror("Errore", "Launch floor/giorni/prenotazioni devono essere numeri validi.")
            return

        try:
            horizon = int(float(self.daily_horizon.get().strip()))
            horizon = max(14, min(365, horizon))
        except Exception:
            horizon = 90

        # Save config
        gui_cfg = {
            "csv_url": url,
            "go_live": self.go_live.get().strip(),
            "direct_base": base_input,
            "auto_adjust": bool(self.auto_adjust.get()),
            "launch_floor": launch_floor,
            "launch_days": launch_days,
            "launch_bookings": launch_bookings_th,
            "face20": bool(self.face20.get()),
            "airbnb_has20": bool(self.airbnb_has20.get()),
            "booking_has20": bool(self.booking_has20.get()),
            "airbnb_weekly_pct": float(self.airbnb_weekly.get() or 5),
            "booking_nonref_pct": float(self.booking_nonref.get() or 10),
            "booking_weekly_pct": float(self.booking_weekly.get() or 15),
            "daily_horizon": horizon,
        }
        try:
            save_gui_config(gui_cfg)
        except Exception as e:
            messagebox.showwarning("Attenzione", f"Non riesco a salvare gui_config.json: {e}\nSuggerimento: sposta fuori da OneDrive.")

        # Load shape + peaks
        weekend_uplift, month_mult, shape_src = compute_shape_overrides()
        peaks = load_peak_weeks()

        # Market priors
        priors = load_market_priors()
        anchor = compute_market_anchor_from_priors(priors) if priors else None
        target14 = anchor["target_occ_14"] if anchor else 0.55
        target30 = anchor["target_occ_30"] if anchor else 0.40
        benchmark_base_2p = anchor["anchor_base_2p"] if anchor else None

        start_day = go_live_dt if go_live_dt > date.today() else date.today()
        self.last_go_live_month = go_live_dt.month

        self._append("Scarico CSV…")
        try:
            rows = fetch_csv_rows(url)
            schema = infer_schema(rows)
            blocked = build_blocked_nights(rows, schema)
        except Exception as e:
            messagebox.showerror("Errore CSV", str(e))
            return

        b14, t14, occ14 = occ_stats(blocked, start_day, 14)
        b30, t30, occ30 = occ_stats(blocked, start_day, 30)
        future_bookings = count_future_bookings(rows, schema, go_live_dt)

        in_launch = (date.today() < (go_live_dt + timedelta(days=launch_days))) or (future_bookings < launch_bookings_th)

        suggested_base, why = recommend_direct_base(
            current_base=base_input,
            occ14=occ14,
            occ30=occ30,
            target14=target14,
            target30=target30,
            allow_adjust=bool(self.auto_adjust.get()),
            launch_floor=launch_floor,
            in_launch=in_launch
        )
        save_state_base(suggested_base)

        self._append("")
        self._append("=== Shape (stagionalità + weekend) ===")
        self._append(f"Sorgente: {shape_src}")
        self._append(f"Weekend uplift (ven/sab): x{weekend_uplift:.2f}")
        self._append("Month mult (1..12): " + ", ".join([f"{m}:{month_mult[m]:.2f}" for m in range(1,13)]))

        self._append("")
        self._append("=== Peak weeks (eventi) ===")
        self._append(f"Regole: {len(peaks.get('rules', []))} | Overrides: {len(peaks.get('date_overrides', []))}")
        if not peaks.get("rules") and not peaks.get("date_overrides"):
            self._append("Nessun peak_weeks.json trovato → nessun uplift eventi.")

        self._append("")
        self._append("=== Benchmark locale (Airbtics) ===")
        if anchor:
            self._append(f"Anchor 7p: ~{round_eur(anchor['anchor_adr_7p'])}€ | Benchmark base 2p: ~{round_eur(anchor['anchor_base_2p'])}€")
            self._append(f"Target occ 14g: {anchor['target_occ_14']:.0%} | 30g: {anchor['target_occ_30']:.0%}")
        else:
            self._append("market_priors.json non trovato → uso target default (55%/40%).")

        delta_pct = None
        if benchmark_base_2p and benchmark_base_2p > 0:
            delta_pct = (base_input / benchmark_base_2p) - 1.0
            self._append(f"Tu (Direct base 2p): {round_eur(base_input)}€  => delta vs benchmark: {delta_pct:+.0%}")

        self._append("")
        self._append("=== Feedback pacing (da CSV) ===")
        self._append(f"GO-LIVE: {self.go_live.get().strip()} | Pacing da: {start_day.isoformat()}")
        self._append(f"Prenotazioni future (da go-live): {future_bookings}")
        self._append(f"Occ 14 giorni: {b14}/{t14} = {occ14:.0%} (target {target14:.0%})")
        self._append(f"Occ 30 giorni: {b30}/{t30} = {occ30:.0%} (target {target30:.0%})")
        self._append(f"Direct base usato (2p feriale): {round_eur(suggested_base)}€ (tu: {round_eur(base_input)}€)")
        self._append(f"Motivo: {why}")
        self._append(f"Launch mode: {'ON' if in_launch else 'OFF'} (floor {round_eur(launch_floor)}€)")

        matrix, todo = build_month_matrix(
            direct_base=suggested_base,
            face20=bool(self.face20.get()),
            airbnb_has20=bool(self.airbnb_has20.get()),
            booking_has20=bool(self.booking_has20.get()),
            weekend_uplift=weekend_uplift,
            month_mult=month_mult,
        )
        self.last_matrix = matrix

        daily = build_daily_matrix(
            start_day=start_day,
            days=horizon,
            direct_base=suggested_base,
            face20=bool(self.face20.get()),
            airbnb_has20=bool(self.airbnb_has20.get()),
            booking_has20=bool(self.booking_has20.get()),
            weekend_uplift=weekend_uplift,
            month_mult=month_mult,
            peaks=peaks
        )
        self.last_daily = daily

        def show_block(title: str, m: int):
            row = self._row_for_month(m)
            if not row:
                return
            self._append("")
            self._append(f"=== {title} (mese {m}) — 2 ospiti ===")
            self._append(f"SITO Direct: weekday {row['direct_weekday_2p']}  weekend {row['direct_weekend_2p']}")
            self._append(f"AIRBNB:     base    {row['airbnb_base_weekday_2p']}  weekend {row['airbnb_weekend_2p']}")
            self._append(f"BOOKING:    std     {row['booking_std_weekday_2p']}  weekend {row['booking_std_weekend_2p']}")

        show_block("Mese GO-LIVE", go_live_dt.month)
        show_block("Peak Luglio", 7)
        show_block("Peak Agosto", 8)

        self._append("")
        self._append("=== Prossimi 14 giorni (con eventi) ===")
        self._append("date       dow peak  Direct2p Airbnb2p Booking2p  label")
        for row in daily[:14]:
            lab = (row["peak_label"] or "")
            self._append(f"{row['date']}  {row['weekday']:<3} x{row['peak_mult']:<4} {row['direct_2p']:>7}  {row['airbnb_2p']:>7}  {row['booking_2p']:>8}  {lab}")

        checklist = self._build_checklist_text()
        self.last_checklist_text = checklist
        self._append("")
        self._append("=== Checklist (cosa impostare) ===")
        for line in checklist.splitlines():
            self._append(line)

        feedback = {
            "occ14_booked": b14, "occ14_total": t14, "occ14_pct": occ14,
            "occ30_booked": b30, "occ30_total": t30, "occ30_pct": occ30,
            "target14": target14, "target30": target30,
            "benchmark_base_2p": benchmark_base_2p,
            "delta_vs_benchmark_pct": delta_pct,
            "direct_base_input": round_eur(base_input),
            "direct_base_suggested": round_eur(suggested_base),
            "why": why,
            "in_launch": in_launch,
            "launch_floor": round_eur(launch_floor),
            "shape_source": shape_src,
            "weekend_uplift": weekend_uplift,
        }
        config_used = gui_cfg

        try:
            write_outputs(
                go_live=self.go_live.get().strip(),
                pacing_from=start_day.isoformat(),
                feedback=feedback,
                config=config_used,
                matrix=matrix,
                daily=daily,
                todo=todo,
                checklist_text=checklist
            )
        except Exception as e:
            messagebox.showerror("Errore output", f"Non riesco a scrivere i file output: {e}\nSuggerimento: sposta fuori da OneDrive.")
            return

        self._append("")
        self._append("=== Output ===")
        self._append("Creati: recommendations.txt, matrix_monthly.csv, matrix_daily.csv, settings_recap.json")

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
