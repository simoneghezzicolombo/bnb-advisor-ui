from __future__ import annotations

import csv
import json
import os
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from tkinter import ttk, messagebox
from typing import Optional, List, Tuple, Dict

import requests

CONFIG_FILE = "gui_config.json"
STATE_FILE = "advisor_state.json"
MARKET_PRIORS_FILE = "market_priors.json"
INSIDE_SHAPE_FILE = "inside_shape.json"

# ==========================
# DEFAULT RULES (tweak later if you want)
# ==========================

DIRECT_DISCOUNT = 0.15          # Sito sempre -15% vs OTA effective
WEEKEND_UPLIFT = 1.25           # Weekend +25% (ven/sab)
PRICE_STEP_EUR = 5.0
MAX_STEP_PER_RUN = 2            # max 10€ per run

# “Launch mode” (evita che l’algoritmo ti butti giù all’inizio)
LAUNCH_FLOOR_EUR_DEFAULT = 60.0
LAUNCH_DAYS_DEFAULT = 21                 # primi 21 giorni dal go-live
LAUNCH_BOOKINGS_THRESHOLD_DEFAULT = 3    # oppure prime 3 prenotazioni

# Stagionalità graduale (per ora manuale; poi possiamo stimarla anche da dataset)
MONTH_MULT = {
    1: 0.95, 2: 0.95, 3: 1.00, 4: 1.05, 5: 1.10, 6: 1.15,
    7: 1.35, 8: 1.35,
    9: 1.10, 10: 1.00, 11: 0.95, 12: 1.10
}

# Scatti ospiti: li usiamo per collegare Airbtics (7p) al nostro base (2p)
# e per checklist piattaforme
GUEST_STEP_3_5 = 15
GUEST_STEP_6_7 = 10

def delta_2p_to_7p_default() -> int:
    # 3-5: +15 * 3 = 45, 6-7: +10 * 2 = 20 => 65
    return 3 * GUEST_STEP_3_5 + 2 * GUEST_STEP_6_7


# ==========================
# HELPERS
# ==========================

def load_inside_shape() -> Optional[dict]:
    if not os.path.exists(INSIDE_SHAPE_FILE):
        return None
    try:
        with open(INSIDE_SHAPE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("combined")
    except Exception:
        return None

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

def load_market_priors() -> Optional[dict]:
    if not os.path.exists(MARKET_PRIORS_FILE):
        return None
    try:
        with open(MARKET_PRIORS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None

def compute_market_anchor_from_priors(priors: dict) -> Optional[dict]:
    """
    Priors file from Airbtics:
    - markets: list of { occ, adr_7p, w }
    - delta_2p_to_7p is 65 by your guest steps
    Output:
    - anchor_adr_7p (weighted)
    - anchor_base_2p (anchor_adr_7p - delta)
    - target_occ_30 (weighted occ)
    - target_occ_14 (slightly higher)
    """
    markets = priors.get("markets", [])
    if not markets:
        return None

    # delta from priors if present, else computed
    delta = priors.get("assumptions", {}).get("delta_2p_to_7p", None)
    if delta is None:
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

    # step-based adjustment vs targets
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

shape = load_inside_shape()
weekend_uplift = float(shape["weekend_uplift"]) if shape and shape.get("weekend_uplift") else WEEKEND_UPLIFT
month_mult = {int(k): float(v) for k, v in (shape.get("month_mult", {}) if shape else {}).items()}
if len(month_mult) == 12:
    # override global multipliers for this run
    for m in range(1, 13):
        MONTH_MULT[m] = month_mult[m]
# override weekend uplift for this run
global WEEKEND_UPLIFT
WEEKEND_UPLIFT = weekend_uplift


def build_month_matrix(
    direct_base: float,
    face20: bool,
    airbnb_has20: bool,
    booking_has20: bool,
) -> Tuple[list[dict], list[str]]:
    """
    Matrix (2 ospiti): Direct weekday/weekend + Airbnb base/weekend + Booking std/weekend.

    Face -20 (no stacking):
    - if face20=True and platform has -20 active => set base higher /0.8
    - else base=OTA effective and we show TODO
    """
    todo = []
    if face20 and not airbnb_has20:
        todo.append("Airbnb: crea una promo -20% quando disponibile (dopo le prime 3 prenotazioni) per mantenere l'effetto sconto.")
    if face20 and not booking_has20:
        todo.append("Booking: attiva una promo/deal -20% per mantenere l'effetto sconto (senza stacking).")

    ab_factor = 0.8 if (face20 and airbnb_has20) else 1.0
    bk_factor = 0.8 if (face20 and booking_has20) else 1.0

    matrix = []
    for m in range(1, 13):
        mult = MONTH_MULT[m]
        direct_weekday = direct_base * mult
        direct_weekend = direct_base * mult * WEEKEND_UPLIFT

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

def write_outputs(
    go_live: str,
    pacing_from: str,
    feedback: dict,
    config: dict,
    matrix: list[dict],
    todo: list[str],
    checklist_text: str
) -> None:
    # matrix csv
    with open("matrix_monthly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "month","direct_weekday_2p","direct_weekend_2p",
            "airbnb_base_weekday_2p","airbnb_weekend_2p",
            "booking_std_weekday_2p","booking_std_weekend_2p"
        ])
        w.writeheader()
        for row in matrix:
            w.writerow(row)

    recap = {
        "go_live_date": go_live,
        "pacing_from": pacing_from,
        "feedback": feedback,
        "config_used": config,
        "todo": todo,
        "matrix_monthly_2p": matrix,
        "checklist": checklist_text
    }
    with open("settings_recap.json", "w", encoding="utf-8") as f:
        json.dump(recap, f, ensure_ascii=False, indent=2)

    # recommendations txt
    lines = []
    lines.append("=== RECOMMENDATIONS ===")
    lines.append(f"GO-LIVE: {go_live}")
    lines.append(f"Pacing calcolato da: {pacing_from}")
    lines.append("")
    for k, v in feedback.items():
        if k.startswith("_"):
            continue
    lines.append(f"Occ 14 giorni: {feedback['occ14_booked']}/{feedback['occ14_total']} = {feedback['occ14_pct']:.0%} (target {feedback['target14']:.0%})")
    lines.append(f"Occ 30 giorni: {feedback['occ30_booked']}/{feedback['occ30_total']} = {feedback['occ30_pct']:.0%} (target {feedback['target30']:.0%})")
    if feedback.get("benchmark_base_2p") is not None:
        lines.append(f"Benchmark locale (Airbtics) base 2p: ~{round_eur(feedback['benchmark_base_2p'])}€")
        lines.append(f"Tu (Direct base 2p): {feedback['direct_base_input']}€  => delta vs benchmark: {feedback['delta_vs_benchmark_pct']:+.0%}")
    lines.append("")
    lines.append(f"Direct base usato (2p feriale): {feedback['direct_base_suggested']}€ (tu hai inserito: {feedback['direct_base_input']}€)")
    lines.append(f"Motivo: {feedback['why']}")
    if feedback.get("in_launch"):
        lines.append(f"Launch mode: ON (floor {feedback['launch_floor']}€)")
    lines.append("")
    lines.append("Promo selezionate:")
    lines.append(f"- Face -20%: {config['face20']}")
    lines.append(f"- Airbnb -20 attivo ora: {config['airbnb_has20']}")
    lines.append(f"- Booking -20 attivo ora: {config['booking_has20']}")
    lines.append(f"- Airbnb weekly %: {config['airbnb_weekly_pct']}")
    lines.append(f"- Booking nonref %: {config['booking_nonref_pct']}")
    lines.append(f"- Booking weekly %: {config['booking_weekly_pct']}")
    lines.append("")
    if todo:
        lines.append("TODO (per mantenere -20 senza stacking):")
        for t in todo:
            lines.append(f"- {t}")
        lines.append("")
    lines.append("CHECKLIST (cosa impostare):")
    lines.append(checklist_text)
    lines.append("")
    lines.append("File generati: matrix_monthly.csv, settings_recap.json")

    with open("recommendations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ==========================
# GUI
# ==========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BnB Advisor — GUI (Airbtics + CSV)")
        self.geometry("1020x760")

        cfg = load_gui_config()

        self.csv_url = tk.StringVar(value=cfg.get("csv_url", ""))
        self.go_live = tk.StringVar(value=cfg.get("go_live", "2026-02-10"))

        state_base = load_state_default_base(60.0)
        self.direct_base = tk.StringVar(value=str(cfg.get("direct_base", state_base)))

        self.auto_adjust = tk.BooleanVar(value=cfg.get("auto_adjust", True))

        # Launch controls
        self.launch_floor = tk.StringVar(value=str(cfg.get("launch_floor", LAUNCH_FLOOR_EUR_DEFAULT)))
        self.launch_days = tk.StringVar(value=str(cfg.get("launch_days", LAUNCH_DAYS_DEFAULT)))
        self.launch_bookings = tk.StringVar(value=str(cfg.get("launch_bookings", LAUNCH_BOOKINGS_THRESHOLD_DEFAULT)))

        # Promotions
        self.face20 = tk.BooleanVar(value=cfg.get("face20", True))
        self.airbnb_has20 = tk.BooleanVar(value=cfg.get("airbnb_has20", True))
        self.booking_has20 = tk.BooleanVar(value=cfg.get("booking_has20", True))

        self.airbnb_weekly = tk.StringVar(value=str(cfg.get("airbnb_weekly_pct", 5)))
        self.booking_nonref = tk.StringVar(value=str(cfg.get("booking_nonref_pct", 5)))
        self.booking_weekly = tk.StringVar(value=str(cfg.get("booking_weekly_pct", 5)))

        self.last_matrix: Optional[list[dict]] = None
        self.last_go_live_month: Optional[int] = None
        self.last_checklist_text: str = ""

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        ttk.Label(frm, text="CSV URL (Google Sheets pubblicato come CSV):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.csv_url, width=125).grid(row=1, column=0, columnspan=6, sticky="we")

        ttk.Label(frm, text="Go-live (YYYY-MM-DD):").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.go_live, width=16).grid(row=2, column=1, sticky="w")

        ttk.Label(frm, text="Direct base 2 ospiti feriale €:").grid(row=2, column=2, sticky="w")
        ttk.Entry(frm, textvariable=self.direct_base, width=8).grid(row=2, column=3, sticky="w")

        ttk.Checkbutton(frm, text="Auto-adjust (pacing)", variable=self.auto_adjust).grid(row=2, column=4, sticky="w")

        ttk.Separator(frm).grid(row=3, column=0, columnspan=6, sticky="we", pady=8)

        # Launch settings
        ttk.Label(frm, text="Launch floor €:").grid(row=4, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.launch_floor, width=8).grid(row=4, column=1, sticky="w")

        ttk.Label(frm, text="Launch giorni:").grid(row=4, column=2, sticky="w")
        ttk.Entry(frm, textvariable=self.launch_days, width=6).grid(row=4, column=3, sticky="w")

        ttk.Label(frm, text="Launch prenotazioni:").grid(row=4, column=4, sticky="w")
        ttk.Entry(frm, textvariable=self.launch_bookings, width=6).grid(row=4, column=5, sticky="w")

        ttk.Separator(frm).grid(row=5, column=0, columnspan=6, sticky="we", pady=8)

        # Promo toggles
        ttk.Checkbutton(frm, text="Face -20% (senza stacking)", variable=self.face20).grid(row=6, column=0, sticky="w")
        ttk.Checkbutton(frm, text="Airbnb: -20% attivo ora", variable=self.airbnb_has20).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Booking: -20% attivo ora", variable=self.booking_has20).grid(row=6, column=2, sticky="w")

        ttk.Label(frm, text="Airbnb weekly %:").grid(row=7, column=0, sticky="w")
        ttk.Spinbox(frm, from_=0, to=30, textvariable=self.airbnb_weekly, width=6).grid(row=7, column=1, sticky="w")

        ttk.Label(frm, text="Booking nonref %:").grid(row=7, column=2, sticky="w")
        ttk.Spinbox(frm, from_=0, to=30, textvariable=self.booking_nonref, width=6).grid(row=7, column=3, sticky="w")

        ttk.Label(frm, text="Booking weekly %:").grid(row=7, column=4, sticky="w")
        ttk.Spinbox(frm, from_=0, to=30, textvariable=self.booking_weekly, width=6).grid(row=7, column=5, sticky="w")

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=8, column=0, columnspan=6, sticky="w", pady=10)
        ttk.Button(btns, text="Run advisor", command=self.run).pack(side="left", padx=6)
        ttk.Button(btns, text="Apri cartella output", command=self.open_folder).pack(side="left", padx=6)
        ttk.Button(btns, text="Copia GO-LIVE month", command=self.copy_go_live_month).pack(side="left", padx=6)
        ttk.Button(btns, text="Copia checklist", command=self.copy_checklist).pack(side="left", padx=6)

        ttk.Separator(frm).grid(row=9, column=0, columnspan=6, sticky="we", pady=8)

        self.out = tk.Text(frm, height=24, wrap="word")
        self.out.grid(row=10, column=0, columnspan=6, sticky="nsew")
        frm.rowconfigure(10, weight=1)
        frm.columnconfigure(0, weight=1)

        ttk.Label(
            frm,
            text="Nota: CSV URL e settaggi vengono salvati in gui_config.json. Benchmark locale letto da market_priors.json."
        ).grid(row=11, column=0, columnspan=6, sticky="w")

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
        bn = int(round(float(self.booking_nonref.get() or "5")))
        bw = int(round(float(self.booking_weekly.get() or "5")))
        face20 = self.face20.get()

        delta = delta_2p_to_7p_default()
        guest_steps = f"3–5: +{GUEST_STEP_3_5}€/ospite | 6–7: +{GUEST_STEP_6_7}€/ospite (2→7 = +{delta}€)"

        lines = []
        lines.append("SITO (Direct)")
        lines.append(f"- Regola: Direct sempre -{int(DIRECT_DISCOUNT*100)}% vs OTA effective")
        lines.append("- Usa i valori mese GO-LIVE + Peak (Lug/Ago) dalla matrice")
        lines.append("")
        lines.append("AIRBNB")
        lines.append("- Smart Pricing: OFF (se vuoi usare Weekend price)")
        lines.append("- Base price (dom–gio): usa 'AIRBNB base' della matrice per il mese/range")
        lines.append("- Weekend price (ven–sab): usa 'AIRBNB weekend' della matrice per il mese/range")
        lines.append(f"- Weekly discount: {aw}%")
        lines.append(f"- Extra guests (oltre 2): usa gli scatti → {guest_steps}")
        lines.append(f"- Face -20%: {'ON' if face20 else 'OFF'} (NO stacking: se c’è già -20, non aggiungere un altro -20)")
        lines.append("")
        lines.append("BOOKING")
        lines.append("- Standard Rate weekday/weekend: usa 'BOOKING std' della matrice per il mese/range")
        lines.append(f"- Non-refundable: {bn}% più economico dello Standard")
        lines.append(f"- Weekly plan: {bw}% più economico dello Standard")
        lines.append(f"- Pricing per ospite: usa gli scatti → {guest_steps}")
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
            aw = float(self.airbnb_weekly.get().strip().replace(",", "."))
            bn = float(self.booking_nonref.get().strip().replace(",", "."))
            bw = float(self.booking_weekly.get().strip().replace(",", "."))
        except Exception:
            messagebox.showerror("Errore", "Le percentuali devono essere numeri (es: 5).")
            return

        try:
            launch_floor = float(self.launch_floor.get().strip().replace(",", "."))
            launch_days = int(float(self.launch_days.get().strip()))
            launch_bookings_th = int(float(self.launch_bookings.get().strip()))
        except Exception:
            messagebox.showerror("Errore", "Launch floor/giorni/prenotazioni devono essere numeri validi.")
            return

        # Save GUI config
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
            "airbnb_weekly_pct": aw,
            "booking_nonref_pct": bn,
            "booking_weekly_pct": bw,
        }
        try:
            save_gui_config(gui_cfg)
        except Exception as e:
            messagebox.showwarning(
                "Attenzione",
                f"Non riesco a salvare gui_config.json: {e}\nConsiglio: sposta la cartella fuori da OneDrive (es. C:\\bnb_advisor)."
            )

        # Market priors (Airbtics)
        priors = load_market_priors()
        anchor = compute_market_anchor_from_priors(priors) if priors else None

        # Targets: if priors exist, use dynamic targets; else fallback to safe defaults
        target14 = anchor["target_occ_14"] if anchor else 0.55
        target30 = anchor["target_occ_30"] if anchor else 0.40
        benchmark_base_2p = anchor["anchor_base_2p"] if anchor else None

        # Pacing period starts at go-live (or today if go-live passed)
        start_day = go_live_dt if go_live_dt > date.today() else date.today()
        self.last_go_live_month = go_live_dt.month

        # Read bookings from CSV
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

        # Launch mode conditions
        in_launch = (date.today() < (go_live_dt + timedelta(days=launch_days))) or (future_bookings < launch_bookings_th)

        # Recommend base
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

        # Output: benchmark + pacing
        self._append("")
        self._append("=== Benchmark locale (Airbtics) ===")
        if anchor:
            self._append(f"Anchor 7p (media pesata): ~{round_eur(anchor['anchor_adr_7p'])}€")
            self._append(f"Delta 2p→7p (scatti): {round_eur(anchor['delta_2p_to_7p'])}€")
            self._append(f"Benchmark base 2p (weekday): ~{round_eur(anchor['anchor_base_2p'])}€")
            self._append(f"Target occupancy 14g: {anchor['target_occ_14']:.0%} | 30g: {anchor['target_occ_30']:.0%}")
            used_names = ", ".join([u["name"] for u in anchor["used_markets"]])
            self._append(f"Mercati usati: {used_names}")
        else:
            self._append("market_priors.json non trovato o non valido → uso target default (55%/40%).")

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
        self._append(f"Direct base usato (2p feriale): {round_eur(suggested_base)}€ (tu hai inserito: {round_eur(base_input)}€)")
        self._append(f"Motivo: {why}")
        if in_launch:
            self._append(f"Launch mode: ON → floor {round_eur(launch_floor)}€ (evita di scendere troppo all’inizio)")
        else:
            self._append("Launch mode: OFF")

        # Build monthly matrix
        matrix, todo = build_month_matrix(
            suggested_base,
            bool(self.face20.get()),
            bool(self.airbnb_has20.get()),
            bool(self.booking_has20.get())
        )
        self.last_matrix = matrix

        def show_block(title: str, m: int):
            row = self._row_for_month(m)
            if not row:
                return
            self._append("")
            self._append(f"=== {title} (mese {m}) — numeri 2 ospiti ===")
            self._append(f"SITO Direct: weekday {row['direct_weekday_2p']}  weekend {row['direct_weekend_2p']}")
            self._append(f"AIRBNB:     base    {row['airbnb_base_weekday_2p']}  weekend {row['airbnb_weekend_2p']}")
            self._append(f"BOOKING:    std     {row['booking_std_weekday_2p']}  weekend {row['booking_std_weekend_2p']}")

        cur_m = date.today().month
        gl_m = go_live_dt.month

        show_block("Mese corrente", cur_m)
        show_block("Mese GO-LIVE", gl_m)
        show_block("Peak Luglio", 7)
        show_block("Peak Agosto", 8)

        # Checklist
        checklist = self._build_checklist_text()
        self.last_checklist_text = checklist
        self._append("")
        self._append("=== Checklist (cosa impostare) ===")
        for line in checklist.splitlines():
            self._append(line)

        # Write files
        feedback = {
            "occ14_booked": b14, "occ14_total": t14, "occ14_pct": occ14,
            "occ30_booked": b30, "occ30_total": t30, "occ30_pct": occ30,
            "target14": target14,
            "target30": target30,
            "benchmark_base_2p": benchmark_base_2p,
            "delta_vs_benchmark_pct": delta_pct,
            "direct_base_input": round_eur(base_input),
            "direct_base_suggested": round_eur(suggested_base),
            "why": why,
            "in_launch": in_launch,
            "launch_floor": round_eur(launch_floor),
            "_future_bookings": future_bookings
        }
        config_used = {
            "csv_url": url,
            "go_live": self.go_live.get().strip(),
            "auto_adjust": bool(self.auto_adjust.get()),
            "direct_base_input": round_eur(base_input),
            "launch_floor": round_eur(launch_floor),
            "launch_days": launch_days,
            "launch_bookings_threshold": launch_bookings_th,
            "face20": bool(self.face20.get()),
            "airbnb_has20": bool(self.airbnb_has20.get()),
            "booking_has20": bool(self.booking_has20.get()),
            "airbnb_weekly_pct": int(round(aw)),
            "booking_nonref_pct": int(round(bn)),
            "booking_weekly_pct": int(round(bw)),
        }

        try:
            write_outputs(
                go_live=self.go_live.get().strip(),
                pacing_from=start_day.isoformat(),
                feedback=feedback,
                config=config_used,
                matrix=matrix,
                todo=todo,
                checklist_text=checklist
            )
        except Exception as e:
            messagebox.showerror("Errore output", f"Non riesco a scrivere i file output: {e}\nConsiglio: sposta la cartella fuori da OneDrive.")
            return

        self._append("")
        self._append("=== Output ===")
        self._append("Creati: recommendations.txt, matrix_monthly.csv, settings_recap.json")
        if todo:
            self._append("")
            self._append("TODO per mantenere -20 senza stacking:")
            for t in todo:
                self._append(f"- {t}")


def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
