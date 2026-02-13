from __future__ import annotations

import csv
import json
import time
import math
import requests
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List, Tuple

# ==========================
# CONFIG (EDIT THESE)
# ==========================

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTD0-p4QLYCfShcs8elKAz-AV3o8iX5Y8dMMj7QqUGWMXtrkp1fnUfdAL7DHfs_NiWveLdULMoE-kGj/pub?gid=859540155&single=true&output=csv"

# 👉 La tua data reale di partenza (go-live), quella che vuoi comunicare
GO_LIVE_DATE = "2026-02-10"  # <-- cambia qui (YYYY-MM-DD)

# Regola: il sito è sempre -15% rispetto al prezzo effettivo OTA (prima delle service fee)
DIRECT_DISCOUNT = 0.15

# Prezzo di partenza che hai deciso (Direct, 2 ospiti, feriale) — poi l’advisor ti dirà se tenerlo o muoverlo
DIRECT_BASE_WEEKDAY_2P_START = 60.0

# Weekend uplift (semplice)
WEEKEND_UPLIFT = 1.25  # +25%

# Scatti ospiti (come vuoi tu: 15/10)
EXTRA_15 = 15.0  # ospiti 3..5
EXTRA_10 = 10.0  # ospiti 6..7
MAX_GUESTS = 7

# Stagionalità graduale per mese (puoi modificare quando vuoi)
MONTH_MULT = {
    1: 0.95, 2: 0.95, 3: 1.00, 4: 1.05, 5: 1.10, 6: 1.15,
    7: 1.35, 8: 1.35,
    9: 1.10, 10: 1.00, 11: 0.95, 12: 1.10
}

# Target "launch": quanto vorresti avere occupato nel breve
TARGET_OCC_14 = 0.55
TARGET_OCC_30 = 0.40

# Quanto può consigliarti di spostare il prezzo per run (in euro)
PRICE_STEP_EUR = 5.0
MAX_STEP_PER_RUN = 2  # max 10€ per run

# Stato locale (ricorda ultimo base consigliato)
STATE_FILE = "advisor_state.json"

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

def is_weekend(d: date) -> bool:
    return d.weekday() in (4, 5)  # ven/sab

def safe_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("€", "").replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def safe_int(x: str) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s.replace(",", ".")))
    except ValueError:
        return None

def is_cancelled(status: str) -> bool:
    st = (status or "").strip().lower()
    return any(k in st for k in ["cancell", "annull", "void", "refunded", "rimbors"])

def is_active_booking_row(row: dict, status_key: str) -> bool:
    """
    In un foglio prenotazioni, di solito una riga con date valide è una prenotazione.
    Qui escludiamo solo se lo status è chiaramente cancellato.
    """
    status = (row.get(status_key, "") or "").strip()
    if is_cancelled(status):
        return False
    return True

def discount_factor(pct: float) -> float:
    return 1.0 - pct

def fetch_csv_rows(url: str) -> list[dict]:
    # cache-bust
    sep = "&" if "?" in url else "?"
    url2 = f"{url}{sep}_cb={int(time.time())}"

    r = requests.get(url2, timeout=25)
    r.raise_for_status()
    reader = csv.DictReader(r.text.splitlines())
    return list(reader)

def normalize_keys(row: dict) -> dict:
    return {str(k).strip().lower(): v for k, v in row.items()}

def find_key(keys: List[str], candidates: List[str]) -> Optional[str]:
    """
    Trova una colonna nel CSV cercando match "contiene" (case-insensitive).
    """
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

def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"direct_base_weekday_2p": DIRECT_BASE_WEEKDAY_2P_START}

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def guest_extras(guests: int) -> float:
    g = max(1, min(guests, MAX_GUESTS))
    if g <= 2:
        return 0.0
    extra15_count = max(0, min(g, 5) - 2)
    extra10_count = max(0, g - 5)
    return extra15_count * EXTRA_15 + extra10_count * EXTRA_10

def daterange(start: date, end: date):
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=1)

# ==========================
# CORE: READ BOOKINGS
# ==========================

@dataclass
class SheetSchema:
    checkin: str
    checkout: str
    status: str
    platform: Optional[str]
    nights: Optional[str]
    guests: Optional[str]
    booking_date: Optional[str]
    gross: Optional[str]
    net: Optional[str]

def infer_schema(rows: List[dict]) -> SheetSchema:
    if not rows:
        raise SystemExit("CSV vuoto: controlla che il tab pubblicato abbia righe.")

    keys = [k.strip() for k in rows[0].keys()]

    checkin = find_key(keys, ["check-in", "check in", "checkin"])
    checkout = find_key(keys, ["check-out", "check out", "checkout"])
    status = find_key(keys, ["status", "stato"])

    platform = find_key(keys, ["piattaforma", "platform"])
    nights = find_key(keys, ["notti", "nights"])
    guests = find_key(keys, ["ospiti", "guests"])
    booking_date = find_key(keys, ["data prenotazione", "booking date", "data_prenotazione"])
    gross = find_key(keys, ["lordo", "gross"])
    net = find_key(keys, ["netto", "net"])

    if not checkin or not checkout or not status:
        raise SystemExit(
            "Non trovo colonne minime. Servono almeno: Check-in, Check-out, Status "
            "(nomi simili vanno bene)."
        )
    return SheetSchema(
        checkin=checkin,
        checkout=checkout,
        status=status,
        platform=platform,
        nights=nights,
        guests=guests,
        booking_date=booking_date,
        gross=gross,
        net=net,
    )

def build_blocked_nights(rows: List[dict], schema: SheetSchema) -> set[date]:
    blocked: set[date] = set()
    for row in rows:
        if not is_active_booking_row(row, schema.status):
            continue

        ci = parse_date_any(row.get(schema.checkin, ""))
        co = parse_date_any(row.get(schema.checkout, ""))
        if not ci or not co or co <= ci:
            continue

        for d in daterange(ci, co):
            blocked.add(d)
    return blocked

def compute_adr_stats(rows: List[dict], schema: SheetSchema) -> dict:
    """
    Feedback: netto per notte e lordo per notte (se disponibili) per piattaforma.
    """
    by_platform = {}
    for row in rows:
        if not is_active_booking_row(row, schema.status):
            continue

        plat = (row.get(schema.platform, "") if schema.platform else "") or "Sconosciuta"
        plat = str(plat).strip() or "Sconosciuta"

        n_nights = safe_int(row.get(schema.nights, "")) if schema.nights else None
        net = safe_float(row.get(schema.net, "")) if schema.net else None
        gross = safe_float(row.get(schema.gross, "")) if schema.gross else None

        if not n_nights or n_nights <= 0:
            continue

        rec = by_platform.setdefault(plat, {"bookings": 0, "nights": 0, "net_sum": 0.0, "gross_sum": 0.0, "net_cnt": 0, "gross_cnt": 0})
        rec["bookings"] += 1
        rec["nights"] += n_nights
        if net is not None:
            rec["net_sum"] += net
            rec["net_cnt"] += 1
        if gross is not None:
            rec["gross_sum"] += gross
            rec["gross_cnt"] += 1

    # compute adr
    out = {}
    for plat, rec in by_platform.items():
        nights = rec["nights"]
        out[plat] = {
            "bookings": rec["bookings"],
            "nights": nights,
            "net_total": rec["net_sum"] if rec["net_cnt"] else None,
            "gross_total": rec["gross_sum"] if rec["gross_cnt"] else None,
            "net_adr": (rec["net_sum"] / nights) if (rec["net_cnt"] and nights) else None,
            "gross_adr": (rec["gross_sum"] / nights) if (rec["gross_cnt"] and nights) else None,
        }
    return out

def occ_stats(blocked: set[date], start_day: date, horizon_days: int) -> Tuple[int, int, float]:
    total = 0
    booked = 0
    for i in range(horizon_days):
        d = start_day + timedelta(days=i)
        total += 1
        if d in blocked:
            booked += 1
    return booked, total, (booked / total if total else 0.0)

# ==========================
# ADVISOR (pace-based ML)
# ==========================

def recommend_direct_base(current_base: float, occ14: float, occ30: float) -> Tuple[float, str]:
    """
    “ML pratico”: usa pacing (occupazione breve) per consigliare +/-.
    È robusto anche con pochi dati.
    """
    score = 0

    if occ14 < TARGET_OCC_14 - 0.10:
        score -= 2
    elif occ14 < TARGET_OCC_14 - 0.05:
        score -= 1
    elif occ14 > TARGET_OCC_14 + 0.10:
        score += 2
    elif occ14 > TARGET_OCC_14 + 0.05:
        score += 1

    if occ30 < TARGET_OCC_30 - 0.10:
        score -= 1
    elif occ30 > TARGET_OCC_30 + 0.10:
        score += 1

    score = max(-MAX_STEP_PER_RUN, min(MAX_STEP_PER_RUN, score))
    new_base = max(35.0, current_base + score * PRICE_STEP_EUR)

    if score == 0:
        why = "Pacing ok → mantieni."
    elif score < 0:
        why = f"Pacing sotto target → abbassa di {abs(score)*PRICE_STEP_EUR:.0f}€."
    else:
        why = f"Pacing sopra target → alza di {abs(score)*PRICE_STEP_EUR:.0f}€."
    return new_base, why

# ==========================
# PROMO STRATEGY
# ==========================

@dataclass
class PromoConfig:
    want_face_20_global: bool
    airbnb_platform_20_active: bool   # -20 (prime 3)
    booking_platform_20_active: bool  # -20 (primi mesi)

    airbnb_weekly_pct: float          # always-on promo (>=7 notti)
    booking_nonref_pct: float
    booking_weekly_pct: float

def choose_effective_face_discount(want_global: bool, platform_has_20: bool) -> float:
    """
    Se vuoi “-20 di facciata”:
    - se la piattaforma ha già -20, usiamo quello (non aggiungiamo nulla)
    - se no, consigliamo di impostare un -20 (dove possibile)
    """
    if not want_global:
        return 0.0
    return 0.20

def build_settings(
    direct_base_weekday_2p: float,
    promo: PromoConfig,
    start_day: date
) -> dict:
    """
    Produce matrix mensile + valori pratici da impostare.
    Coerenza: sito sempre -15% vs OTA effective.
    """
    settings = {
        "go_live_date": GO_LIVE_DATE,
        "start_day_used_for_pacing": start_day.isoformat(),
        "rule": "Sito (Direct) sempre -15% rispetto al prezzo effettivo OTA.",
        "guest_steps": "3–5: +15€/ospite | 6–7: +10€/ospite",
        "airbnb": {},
        "booking": {},
        "matrix_monthly_2p": [],
        "warnings": [],
        "todo": []
    }

    # “-20 di facciata” (non stacked)
    face20_airbnb = choose_effective_face_discount(promo.want_face_20_global, promo.airbnb_platform_20_active)
    face20_booking = choose_effective_face_discount(promo.want_face_20_global, promo.booking_platform_20_active)

    # Se vuoi -20 globale ma la piattaforma NON ha promo -20 attiva, ti segnaliamo che dovrai creare una promo (quando possibile)
    if promo.want_face_20_global and not promo.airbnb_platform_20_active:
        settings["todo"].append("Airbnb: crea una promozione -20% (quando disponibile) per mantenere l'effetto 'sempre sconto'.")
    if promo.want_face_20_global and not promo.booking_platform_20_active:
        settings["todo"].append("Booking: attiva una promozione/deal -20% (se vuoi l'effetto 'sempre sconto').")

    # Fattori piattaforma (solo il -20 visibile, senza stacking)
    airbnb_factor = discount_factor(face20_airbnb)  # 0.8 se -20, altrimenti 1
    booking_factor = discount_factor(face20_booking)

    # Always-on promos (long stays / rate plans)
    settings["airbnb"]["weekly_discount_pct"] = int(round(promo.airbnb_weekly_pct * 100))
    settings["airbnb"]["extra_guest_fee_over_2"] = 15  # un solo valore (semplice)
    settings["booking"]["nonref_pct"] = int(round(promo.booking_nonref_pct * 100))
    settings["booking"]["weekly_pct"] = int(round(promo.booking_weekly_pct * 100))
    settings["booking"]["note_guest_pricing"] = "Consiglio: usa pricing per ospite (base 2) e replica scatti 15/10."

    # Matrix per mese (2 ospiti): Direct + Airbnb base/weekend + Booking standard/weekend
    for m in range(1, 13):
        mult = MONTH_MULT[m]

        direct_weekday = direct_base_weekday_2p * mult
        direct_weekend = direct_base_weekday_2p * mult * WEEKEND_UPLIFT

        ota_eff_weekday = direct_weekday / (1 - DIRECT_DISCOUNT)
        ota_eff_weekend = direct_weekend / (1 - DIRECT_DISCOUNT)

        # Base da impostare sulle piattaforme per ottenere ota_eff dopo lo sconto visibile
        airbnb_base_weekday = ota_eff_weekday / airbnb_factor
        airbnb_base_weekend = ota_eff_weekend / airbnb_factor

        booking_std_weekday = ota_eff_weekday / booking_factor
        booking_std_weekend = ota_eff_weekend / booking_factor

        settings["matrix_monthly_2p"].append({
            "month": m,
            "direct_weekday_2p": round_eur(direct_weekday),
            "direct_weekend_2p": round_eur(direct_weekend),
            "airbnb_base_weekday_2p": round_eur(airbnb_base_weekday),
            "airbnb_weekend_2p": round_eur(airbnb_base_weekend),
            "booking_std_weekday_2p": round_eur(booking_std_weekday),
            "booking_std_weekend_2p": round_eur(booking_std_weekend),
        })

    # Warnings: verifica che gli sconti “sempre attivi” non battano il Direct sui soggiorni lunghi
    # Condizione safe: OTA_eff * (1 - extra_discount) >= Direct  => extra_discount <= 15%
    # Quindi: weekly/nonref/weekly plan dovrebbero stare <= 10% quando vuoi essere tranquillo.
    if promo.airbnb_weekly_pct > 0.10:
        settings["warnings"].append("Airbnb: weekly >10% può avvicinarsi troppo al Direct sui soggiorni lunghi (consiglio 5–10%).")
    if promo.booking_nonref_pct > 0.10:
        settings["warnings"].append("Booking: non-refundable >10% può avvicinarsi troppo al Direct (consiglio 5–10%).")
    if promo.booking_weekly_pct > 0.10:
        settings["warnings"].append("Booking: weekly plan >10% può avvicinarsi troppo al Direct (consiglio 5–10%).")

    return settings

# ==========================
# OUTPUT
# ==========================

def write_recommendations_txt(path: str, feedback: dict, settings: dict) -> None:
    lines = []
    lines.append("=== BnB Advisor — Recommendations ===")
    lines.append(f"GO-LIVE (data di partenza): {settings['go_live_date']}")
    lines.append(f"Data usata per pacing: {settings['start_day_used_for_pacing']}")
    lines.append("")
    lines.append("FEEDBACK (pacing):")
    lines.append(f"- Occupazione 14 giorni: {feedback['occ14_booked']}/{feedback['occ14_total']} = {feedback['occ14']:.0%} (target {TARGET_OCC_14:.0%})")
    lines.append(f"- Occupazione 30 giorni: {feedback['occ30_booked']}/{feedback['occ30_total']} = {feedback['occ30']:.0%} (target {TARGET_OCC_30:.0%})")
    lines.append(f"- Consiglio Direct base (2p feriale): {feedback['direct_base_suggested']}€ (prima: {feedback['direct_base_prev']}€)")
    lines.append(f"- Motivo: {feedback['why']}")
    lines.append("")
    lines.append("REGOLE:")
    lines.append(f"- Direct sempre -{int(DIRECT_DISCOUNT*100)}% rispetto al prezzo effettivo OTA")
    lines.append(f"- Weekend uplift: +{int((WEEKEND_UPLIFT-1)*100)}%")
    lines.append(f"- Scatti ospiti: {settings['guest_steps']}")
    lines.append("")
    lines.append("AIRBNB (da impostare):")
    lines.append(f"- Extra guest fee oltre 2: {settings['airbnb']['extra_guest_fee_over_2']}€")
    lines.append(f"- Weekly discount: {settings['airbnb']['weekly_discount_pct']}%")
    lines.append("")
    lines.append("BOOKING (da impostare):")
    lines.append(f"- Non-refundable: {settings['booking']['nonref_pct']}%")
    lines.append(f"- Weekly plan: {settings['booking']['weekly_pct']}%")
    lines.append(f"- Nota: {settings['booking']['note_guest_pricing']}")
    lines.append("")
    if settings["todo"]:
        lines.append("TODO (per mantenere '-20 di facciata' senza stacking):")
        for t in settings["todo"]:
            lines.append(f"- {t}")
        lines.append("")
    if settings["warnings"]:
        lines.append("WARNINGS:")
        for w in settings["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("MATRICE MENSILE: vedi matrix_monthly.csv (weekday/weekend per mese).")

    # ADR feedback (se presente)
    if feedback.get("adr"):
        lines.append("")
        lines.append("STORICO (se disponibile nel foglio):")
        for plat, rec in feedback["adr"].items():
            net_adr = rec.get("net_adr")
            gross_adr = rec.get("gross_adr")
            lines.append(f"- {plat}: prenotazioni={rec['bookings']}, notti={rec['nights']}, netto/nt={net_adr:.2f}€" if net_adr else f"- {plat}: prenotazioni={rec['bookings']}, notti={rec['nights']}")
            if gross_adr:
                lines.append(f"  lordo/nt={gross_adr:.2f}€")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def write_matrix_csv(path: str, matrix: list[dict]) -> None:
    headers = ["month","direct_weekday_2p","direct_weekend_2p","airbnb_base_weekday_2p","airbnb_weekend_2p","booking_std_weekday_2p","booking_std_weekend_2p"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in matrix:
            w.writerow(row)

# ==========================
# MAIN
# ==========================

def main():
    if "INCOLLA_QUI" in SHEET_CSV_URL:
        raise SystemExit("incolla qui")

    go_live = parse_yyyy_mm_dd(GO_LIVE_DATE)
    today = date.today()
    start_day = go_live if go_live > today else today

    # Load state (last base)
    state = load_state()
    current_base = float(state.get("direct_base_weekday_2p", DIRECT_BASE_WEEKDAY_2P_START))

    # Read sheet (read-only)
    rows = fetch_csv_rows(SHEET_CSV_URL)
    schema = infer_schema(rows)

    blocked = build_blocked_nights(rows, schema)
    adr = compute_adr_stats(rows, schema)

    # Pacing (from start_day, not necessarily today)
    b14, t14, occ14 = occ_stats(blocked, start_day, 14)
    b30, t30, occ30 = occ_stats(blocked, start_day, 30)

    # Recommend new base
    new_base, why = recommend_direct_base(current_base, occ14, occ30)

    # Ask promo strategy (intuitive)
    print("\n=== BnB Advisor (on-demand) ===")
    print(f"GO-LIVE impostato: {GO_LIVE_DATE}")
    print(f"Pacing calcolato da: {start_day.isoformat()}\n")

    print(f"Occupazione prossimi 14 giorni: {b14}/{t14} = {occ14:.0%} (target {TARGET_OCC_14:.0%})")
    print(f"Occupazione prossimi 30 giorni: {b30}/{t30} = {occ30:.0%} (target {TARGET_OCC_30:.0%})")
    print(f"Consiglio Direct base (2p feriale): {round_eur(new_base)}€ (prima: {round_eur(current_base)}€)")
    print(f"Motivo: {why}\n")

    want_face_20 = input("Vuoi mantenere un '-20% di facciata' (senza stacking) dove possibile? [Y]: ").strip().lower()
    want_face_20 = True if want_face_20 in ("", "y", "s") else False

    airbnb_platform_20 = input("Airbnb: promo -20 attiva adesso (prime 3)? [Y]: ").strip().lower()
    airbnb_platform_20 = True if airbnb_platform_20 in ("", "y", "s") else False

    booking_platform_20 = input("Booking: promo -20 attiva adesso (primi mesi)? [Y]: ").strip().lower()
    booking_platform_20 = True if booking_platform_20 in ("", "y", "s") else False

    # Always-on promos (keep it simple)
    aw = input("Airbnb: weekly discount sempre attivo (consiglio 5) [%] [5]: ").strip()
    airbnb_weekly = (5.0 if aw == "" else float(aw.replace(",", "."))) / 100.0

    bn = input("Booking: non-refundable % (consiglio 5) [5]: ").strip()
    booking_nonref = (5.0 if bn == "" else float(bn.replace(",", "."))) / 100.0

    bw = input("Booking: weekly plan % (consiglio 5) [5]: ").strip()
    booking_weekly = (5.0 if bw == "" else float(bw.replace(",", "."))) / 100.0

    promo = PromoConfig(
        want_face_20_global=want_face_20,
        airbnb_platform_20_active=airbnb_platform_20,
        booking_platform_20_active=booking_platform_20,
        airbnb_weekly_pct=airbnb_weekly,
        booking_nonref_pct=booking_nonref,
        booking_weekly_pct=booking_weekly,
    )

    settings = build_settings(new_base, promo, start_day)

    # Save outputs
    feedback = {
        "occ14_booked": b14,
        "occ14_total": t14,
        "occ14": occ14,
        "occ30_booked": b30,
        "occ30_total": t30,
        "occ30": occ30,
        "direct_base_prev": round_eur(current_base),
        "direct_base_suggested": round_eur(new_base),
        "why": why,
        "adr": adr,
    }

    with open("settings_recap.json", "w", encoding="utf-8") as f:
        json.dump({"feedback": feedback, "settings": settings}, f, ensure_ascii=False, indent=2)

    write_matrix_csv("matrix_monthly.csv", settings["matrix_monthly_2p"])
    write_recommendations_txt("recommendations.txt", feedback, settings)

    # Update state
    state["direct_base_weekday_2p"] = float(new_base)
    save_state(state)

    print("\n✅ Creati i file:")
    print("- recommendations.txt  (cosa impostare, spiegato)")
    print("- matrix_monthly.csv   (tabella mensile weekday/weekend)")
    print("- settings_recap.json  (tutto in json)")
    print("\nApri recommendations.txt per la guida più facile.")

if __name__ == "__main__":
    main()
