from __future__ import annotations

import os
import io
from datetime import date, datetime, timedelta
import pandas as pd
import streamlit as st

# ---- Import backend: NON tocchiamo algoritmo, solo wrapper UI ----
BACKEND = None
IMPORT_ERRORS = []

for mod in ("bnb_advisor_gui_v3", "bnb_advisor_gui", "bnb_advisor"):
    try:
        BACKEND = __import__(mod)
        BACKEND_NAME = mod
        break
    except Exception as e:
        IMPORT_ERRORS.append(f"{mod}: {e}")

if BACKEND is None:
    st.error("Non trovo il backend (bnb_advisor_gui_v3 / bnb_advisor_gui / bnb_advisor).")
    st.code("\n".join(IMPORT_ERRORS))
    st.stop()

# ---- Helpers UI ----
def _to_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("€", "").replace(",", ".")
        return float(s)
    except:
        return default

def _round_to(x: float, step: int | None):
    if step is None:
        return x
    return round(x / step) * step

def compress_blocks(daily_rows, price_key: str, round_step: int | None = None):
    """
    Converte prezzi giornalieri in blocchi consecutivi:
    Start, End, Notti, Prezzo
    """
    blocks = []
    cur = None

    def close_cur():
        nonlocal cur
        if cur:
            blocks.append(cur)
            cur = None

    for r in daily_rows:
        d = r.get("date")
        p = _to_float(r.get(price_key))
        if p is None:
            continue
        p = _round_to(p, round_step)

        label = (r.get("peak_label") or "").strip()
        dow = (r.get("weekday") or "").strip()

        if cur is None:
            cur = {
                "start": d,
                "end": d,
                "nights": 1,
                "price": int(p) if float(p).is_integer() else p,
                "labels": set([label]) if label else set(),
                "dows": set([dow]) if dow else set(),
            }
            continue

        # consecutivo?
        prev_end = datetime.fromisoformat(cur["end"]).date()
        this_day = datetime.fromisoformat(d).date()
        is_next = (this_day == prev_end + timedelta(days=1))

        if is_next and cur["price"] == (int(p) if float(p).is_integer() else p):
            cur["end"] = d
            cur["nights"] += 1
            if label:
                cur["labels"].add(label)
            if dow:
                cur["dows"].add(dow)
        else:
            close_cur()
            cur = {
                "start": d,
                "end": d,
                "nights": 1,
                "price": int(p) if float(p).is_integer() else p,
                "labels": set([label]) if label else set(),
                "dows": set([dow]) if dow else set(),
            }

    close_cur()

    # finalizza labels
    for b in blocks:
        b["labels"] = ", ".join(sorted([x for x in b["labels"] if x]))
        b["dows"] = ", ".join(sorted([x for x in b["dows"] if x]))
    return blocks

def make_instructions_airbnb(cfg, feedback, blocks):
    weekly = cfg.get("airbnb_weekly_pct", 5)
    lines = []
    lines.append("AIRBNB — cosa fare (manuale, veloce)")
    lines.append("1) Disattiva Smart Pricing (se attivo).")
    lines.append(f"2) Imposta Weekly discount: {weekly:.0f}%.")
    lines.append("3) Imposta Extra guest fees (una volta sola):")
    lines.append("   - 3–5 ospiti: +15€/ospite")
    lines.append("   - 6–7 ospiti: +10€/ospite")
    if cfg.get("face20", True):
        lines.append("4) Promo visibile -20%: ON (evita stacking).")
        if not cfg.get("airbnb_has20", True):
            lines.append("   - TODO: attivare -20% su Airbnb (per evitare doppi sconti).")
    lines.append("")
    lines.append("5) Prezzi calendario (2 ospiti):")
    lines.append("   - Vai su Calendario → seleziona blocchi di date → Modifica prezzo notte")
    lines.append("   - Usa questi blocchi (prezzi già finali):")
    for b in blocks[:60]:
        if b["start"] == b["end"]:
            lines.append(f"   • {b['start']} → {b['price']}€" + (f"  [{b['labels']}]" if b["labels"] else ""))
        else:
            lines.append(f"   • {b['start']} → {b['end']} ({b['nights']} notti) → {b['price']}€" + (f"  [{b['labels']}]" if b["labels"] else ""))
    lines.append("")
    lines.append(f"Base suggerito Direct (controllo pacing): {feedback['direct_base_suggested']}€ — motivo: {feedback['why']}")
    return "\n".join(lines)

def make_instructions_booking(cfg, feedback, blocks):
    nonref = cfg.get("booking_nonref_pct", 5)
    weekly = cfg.get("booking_weekly_pct", 5)
    lines = []
    lines.append("BOOKING — cosa fare (manuale, veloce)")
    lines.append("1) Rate plan:")
    lines.append("   - Standard = prezzi qui sotto")
    lines.append(f"   - Non-refundable: {nonref:.0f}% più economico dello Standard")
    lines.append(f"   - Weekly: {weekly:.0f}% più economico dello Standard")
    lines.append("2) Pricing per ospite (una volta sola):")
    lines.append("   - 3–5 ospiti: +15€/ospite")
    lines.append("   - 6–7 ospiti: +10€/ospite")
    if cfg.get("face20", True) and not cfg.get("booking_has20", True):
        lines.append("3) TODO: attivare promo -20% su Booking (per evitare stacking).")
    lines.append("")
    lines.append("4) Prezzi calendario (Standard, 2 ospiti):")
    lines.append("   - Rates & Availability → Calendar → Bulk edit / multi-selezione date → set price")
    lines.append("   - Usa questi blocchi (prezzi già finali):")
    for b in blocks[:60]:
        if b["start"] == b["end"]:
            lines.append(f"   • {b['start']} → {b['price']}€" + (f"  [{b['labels']}]" if b["labels"] else ""))
        else:
            lines.append(f"   • {b['start']} → {b['end']} ({b['nights']} notti) → {b['price']}€" + (f"  [{b['labels']}]" if b["labels"] else ""))
    lines.append("")
    lines.append(f"Base suggerito Direct (controllo pacing): {feedback['direct_base_suggested']}€ — motivo: {feedback['why']}")
    retu
