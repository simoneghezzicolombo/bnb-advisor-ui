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
    return "\n".join(lines)

def make_instructions_site(feedback):
    return "\n".join([
        "SITO — cosa fare",
        "1) Usa il file prezzi Direct (download sotto).",
        "2) Regola consigliata: Direct ≈ -15% vs OTA effective (già applicato nei valori 'direct_2p').",
        f"3) Base suggerito Direct: {feedback['direct_base_suggested']}€ — motivo: {feedback['why']}",
    ])

def run_engine(cfg: dict) -> dict:
    """
    Wrapper che usa le funzioni del tuo backend (algoritmo invariato).
    Ritorna daily + feedback.
    """
    parse = getattr(BACKEND, "parse_yyyy_mm_dd")
    fetch = getattr(BACKEND, "fetch_csv_rows")
    infer_schema = getattr(BACKEND, "infer_schema")
    build_blocked = getattr(BACKEND, "build_blocked_nights")
    occ_stats = getattr(BACKEND, "occ_stats")
    count_future = getattr(BACKEND, "count_future_bookings")
    recommend = getattr(BACKEND, "recommend_direct_base")
    compute_shape = getattr(BACKEND, "compute_shape_overrides")
    load_peaks = getattr(BACKEND, "load_peak_weeks")
    load_priors = getattr(BACKEND, "load_market_priors")
    anchor_from_priors = getattr(BACKEND, "compute_market_anchor_from_priors")
    build_daily = getattr(BACKEND, "build_daily_matrix")

    url = cfg["csv_url"].strip()
    go_live_dt = parse(cfg["go_live"].strip())
    base_input = float(str(cfg["direct_base"]).replace(",", "."))
    horizon = int(cfg.get("daily_horizon", 90))

    weekend_uplift, month_mult, shape_src = compute_shape()
    peaks = load_peaks()
    priors = load_priors()
    anchor = anchor_from_priors(priors) if priors else None
    target14 = anchor["target_occ_14"] if anchor else 0.55
    target30 = anchor["target_occ_30"] if anchor else 0.40
    benchmark_base_2p = anchor["anchor_base_2p"] if anchor else None

    start_day = go_live_dt if go_live_dt > date.today() else date.today()

    rows = fetch(url)
    schema = infer_schema(rows)
    blocked = build_blocked(rows, schema)

    b14, t14, occ14 = occ_stats(blocked, start_day, 14)
    b30, t30, occ30 = occ_stats(blocked, start_day, 30)
    future_bookings = count_future(rows, schema, go_live_dt)

    launch_floor = float(cfg["launch_floor"])
    launch_days = int(cfg["launch_days"])
    launch_bookings_th = int(cfg["launch_bookings"])

    in_launch = (date.today() < (go_live_dt + timedelta(days=launch_days))) or (future_bookings < launch_bookings_th)

    suggested_base, why = recommend(
        current_base=base_input,
        occ14=occ14,
        occ30=occ30,
        target14=target14,
        target30=target30,
        allow_adjust=bool(cfg["auto_adjust"]),
        launch_floor=launch_floor,
        in_launch=in_launch
    )

    daily = build_daily(
        start_day=start_day,
        days=horizon,
        direct_base=suggested_base,
        face20=bool(cfg["face20"]),
        airbnb_has20=bool(cfg["airbnb_has20"]),
        booking_has20=bool(cfg["booking_has20"]),
        weekend_uplift=weekend_uplift,
        month_mult=month_mult,
        peaks=peaks
    )

    delta_pct = None
    if benchmark_base_2p and benchmark_base_2p > 0:
        delta_pct = (base_input / benchmark_base_2p) - 1.0

    feedback = {
        "pacing_from": start_day.isoformat(),
        "occ14_booked": b14, "occ14_total": t14, "occ14_pct": occ14,
        "occ30_booked": b30, "occ30_total": t30, "occ30_pct": occ30,
        "target14": target14, "target30": target30,
        "benchmark_base_2p": benchmark_base_2p,
        "delta_vs_benchmark_pct": delta_pct,
        "direct_base_input": int(base_input),
        "direct_base_suggested": int(round(suggested_base)),
        "why": why,
        "in_launch": in_launch,
        "launch_floor": int(round(launch_floor)),
        "shape_source": shape_src,
        "weekend_uplift": weekend_uplift,
        "future_bookings": future_bookings,
        "go_live": cfg["go_live"].strip(),
    }

    return {"feedback": feedback, "daily": daily}

# ---------------- UI ----------------
st.set_page_config(page_title="BnB Advisor — Simple UI", layout="wide")
st.title("BnB Advisor — UI super semplice (Airbnb / Booking / Sito)")

with st.sidebar:
    st.subheader("Setup (minimo)")
    csv_url = st.text_input("CSV URL (Google Sheet pubblicato)", value="")
    go_live = st.text_input("Go-live (YYYY-MM-DD)", value=str(date.today()))
    direct_base = st.number_input("Direct base 2p feriale (€)", min_value=10, max_value=2000, value=80, step=1)
    auto_adjust = st.checkbox("Auto-adjust (pacing)", value=True)

    st.divider()
    st.subheader("Promo & Orizzonte")
    face20 = st.checkbox("Promo faccia -20% ON", value=True)
    airbnb_has20 = st.checkbox("Airbnb: -20% già attiva", value=True)
    booking_has20 = st.checkbox("Booking: -20% già attiva", value=True)
    horizon = st.slider("Orizzonte prezzi (giorni)", 14, 365, 90)

    st.divider()
    with st.expander("Avanzate (di solito non tocchi)"):
        launch_floor = st.number_input("Launch floor (€)", min_value=10, max_value=2000, value=60, step=1)
        launch_days = st.number_input("Launch giorni", min_value=1, max_value=120, value=21, step=1)
        launch_bookings = st.number_input("Soglia bookings launch", min_value=0, max_value=20, value=3, step=1)

        airbnb_weekly_pct = st.number_input("Airbnb weekly %", min_value=0, max_value=60, value=5, step=1)
        booking_nonref_pct = st.number_input("Booking non-ref %", min_value=0, max_value=60, value=5, step=1)
        booking_weekly_pct = st.number_input("Booking weekly %", min_value=0, max_value=60, value=5, step=1)

    st.divider()
    round_step = st.selectbox("Blocchi prezzi: raggruppa per", options=[None, 1, 5, 10], index=2,
                             format_func=lambda x: "prezzo identico" if x is None else f"arrotonda a {x}€")
    run = st.button("GENERA PIANO", use_container_width=True)

cfg = {
    "csv_url": csv_url,
    "go_live": go_live,
    "direct_base": direct_base,
    "auto_adjust": auto_adjust,
    "face20": face20,
    "airbnb_has20": airbnb_has20,
    "booking_has20": booking_has20,
    "daily_horizon": horizon,
    "launch_floor": launch_floor if 'launch_floor' in locals() else 60,
    "launch_days": launch_days if 'launch_days' in locals() else 21,
    "launch_bookings": launch_bookings if 'launch_bookings' in locals() else 3,
    "airbnb_weekly_pct": airbnb_weekly_pct if 'airbnb_weekly_pct' in locals() else 5,
    "booking_nonref_pct": booking_nonref_pct if 'booking_nonref_pct' in locals() else 5,
    "booking_weekly_pct": booking_weekly_pct if 'booking_weekly_pct' in locals() else 5,
}

if run:
    if not csv_url.strip().startswith("http"):
        st.error("CSV URL non valido (deve iniziare con http).")
        st.stop()

    with st.spinner("Calcolo in corso…"):
        state = run_engine(cfg)

    st.session_state["state"] = state
    st.session_state["cfg"] = cfg
    st.success("Piano generato.")

state = st.session_state.get("state")
cfg_saved = st.session_state.get("cfg")

if not state:
    st.info("Inserisci CSV + parametri e premi **GENERA PIANO**.")
    st.stop()

feedback = state["feedback"]
daily = state["daily"]
df = pd.DataFrame(daily)

# Header KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("Occ 14g", f"{feedback['occ14_pct']:.0%}", f"target {feedback['target14']:.0%}")
c2.metric("Occ 30g", f"{feedback['occ30_pct']:.0%}", f"target {feedback['target30']:.0%}")
c3.metric("Direct base suggerito", f"{feedback['direct_base_suggested']}€", f"tu: {feedback['direct_base_input']}€")
c4.metric("Launch", "ON" if feedback["in_launch"] else "OFF", f"floor {feedback['launch_floor']}€")

st.caption(f"Motivo: {feedback['why']}")

tab_setup, tab_airbnb, tab_booking, tab_site, tab_files = st.tabs(["Setup", "Airbnb", "Booking", "Sito", "File"])

with tab_setup:
    st.subheader("Anteprima (prime 14 notti)")
    st.dataframe(df.head(14), use_container_width=True)

with tab_airbnb:
    blocks = compress_blocks(daily, "airbnb_2p", round_step=round_step)
    instr = make_instructions_airbnb(cfg_saved, feedback, blocks)

    st.subheader("Istruzioni copiabili")
    st.text_area("Airbnb — copia/incolla", value=instr, height=320)

    st.subheader("Blocchi prezzi (più comodi per inserimento manuale)")
    bdf = pd.DataFrame(blocks)
    st.dataframe(bdf, use_container_width=True, height=360)

    st.subheader("Tabella giorno-per-giorno (2p / 7p)")
    st.dataframe(df[["date","weekday","peak_mult","peak_label","airbnb_2p","airbnb_7p"]].head(60), use_container_width=True)

with tab_booking:
    blocks = compress_blocks(daily, "booking_2p", round_step=round_step)
    instr = make_instructions_booking(cfg_saved, feedback, blocks)

    st.subheader("Istruzioni copiabili")
    st.text_area("Booking — copia/incolla", value=instr, height=320)

    st.subheader("Blocchi prezzi (più comodi per inserimento manuale)")
    bdf = pd.DataFrame(blocks)
    st.dataframe(bdf, use_container_width=True, height=360)

    st.subheader("Tabella giorno-per-giorno (2p / 7p)")
    st.dataframe(df[["date","weekday","peak_mult","peak_label","booking_2p","booking_7p"]].head(60), use_container_width=True)

with tab_site:
    st.subheader("Istruzioni")
    st.text_area("Sito — copia/incolla", value=make_instructions_site(feedback), height=160)

    st.subheader("Prezzi Direct")
    st.dataframe(df[["date","weekday","peak_mult","peak_label","direct_2p","direct_7p"]].head(60), use_container_width=True)

with tab_files:
    st.subheader("Download (senza errori di file aperti)")
    # Daily CSV
    daily_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica matrix_daily.csv", data=daily_csv, file_name="matrix_daily.csv", mime="text/csv")

    # Airbnb blocks
    ablocks = pd.DataFrame(compress_blocks(daily, "airbnb_2p", round_step=round_step))
    st.download_button("Scarica airbnb_blocks.csv", data=ablocks.to_csv(index=False).encode("utf-8"),
                       file_name="airbnb_blocks.csv", mime="text/csv")

    # Booking blocks
    bblocks = pd.DataFrame(compress_blocks(daily, "booking_2p", round_step=round_step))
    st.download_button("Scarica booking_blocks.csv", data=bblocks.to_csv(index=False).encode("utf-8"),
                       file_name="booking_blocks.csv", mime="text/csv")

    # Also save to timestamp folder (optional)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(os.getcwd(), "out", ts)
    os.makedirs(out_dir, exist_ok=True)
    try:
        df.to_csv(os.path.join(out_dir, "matrix_daily.csv"), index=False)
        ablocks.to_csv(os.path.join(out_dir, "airbnb_blocks.csv"), index=False)
        bblocks.to_csv(os.path.join(out_dir, "booking_blocks.csv"), index=False)
        st.success(f"Salvati anche in: {out_dir}")
    except PermissionError:
        st.warning("Non sono riuscito a salvare su disco (file in uso). I download funzionano sempre.")
