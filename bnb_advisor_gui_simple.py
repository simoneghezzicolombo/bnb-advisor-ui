# bnb_advisor_gui_simple.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date, timedelta

# Import dal tuo backend (non tocchiamo l'algoritmo)
from bnb_advisor_gui_v3 import (
    load_gui_config, save_gui_config,
    compute_shape_overrides, load_peak_weeks,
    load_market_priors, compute_market_anchor_from_priors,
    fetch_csv_rows, infer_schema, build_blocked_nights,
    occ_stats, count_future_bookings, recommend_direct_base,
    build_month_matrix, build_daily_matrix, write_outputs,
    parse_yyyy_mm_dd, round_eur,
)

APP_TITLE = "BnB Advisor — SIMPLE (Airbnb / Booking / Sito)"

def run_engine(cfg: dict) -> dict:
    """
    Esegue lo stesso flusso della GUI v3 ma restituisce un dizionario “pulito”
    per la UI semplice.
    """
    url = cfg["csv_url"].strip()
    go_live_dt = parse_yyyy_mm_dd(cfg["go_live"].strip())

    base_input = float(str(cfg["direct_base"]).replace(",", "."))
    horizon = int(cfg.get("daily_horizon", 90))
    horizon = max(14, min(365, horizon))

    # Shape + peaks
    weekend_uplift, month_mult, shape_src = compute_shape_overrides()
    peaks = load_peak_weeks()

    # Market priors / target / benchmark
    priors = load_market_priors()
    anchor = compute_market_anchor_from_priors(priors) if priors else None
    target14 = anchor["target_occ_14"] if anchor else 0.55
    target30 = anchor["target_occ_30"] if anchor else 0.40
    benchmark_base_2p = anchor["anchor_base_2p"] if anchor else None

    start_day = go_live_dt if go_live_dt > date.today() else date.today()

    # CSV → occupancy
    rows = fetch_csv_rows(url)
    schema = infer_schema(rows)
    blocked = build_blocked_nights(rows, schema)

    b14, t14, occ14 = occ_stats(blocked, start_day, 14)
    b30, t30, occ30 = occ_stats(blocked, start_day, 30)
    future_bookings = count_future_bookings(rows, schema, go_live_dt)

    launch_floor = float(cfg["launch_floor"])
    launch_days = int(cfg["launch_days"])
    launch_bookings_th = int(cfg["launch_bookings"])

    in_launch = (date.today() < (go_live_dt + timedelta(days=launch_days))) or (future_bookings < launch_bookings_th)

    suggested_base, why = recommend_direct_base(
        current_base=base_input,
        occ14=occ14,
        occ30=occ30,
        target14=target14,
        target30=target30,
        allow_adjust=bool(cfg["auto_adjust"]),
        launch_floor=launch_floor,
        in_launch=in_launch
    )

    matrix, todo = build_month_matrix(
        direct_base=suggested_base,
        face20=bool(cfg["face20"]),
        airbnb_has20=bool(cfg["airbnb_has20"]),
        booking_has20=bool(cfg["booking_has20"]),
        weekend_uplift=weekend_uplift,
        month_mult=month_mult,
    )

    daily = build_daily_matrix(
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

    checklist = build_checklist_text(cfg)

    feedback = {
        "pacing_from": start_day.isoformat(),
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
        "future_bookings": future_bookings,
        "go_live": cfg["go_live"].strip(),
    }

    # Scrivi gli stessi output standard (così continui ad avere CSV e txt uguali)
    write_outputs(
        go_live=cfg["go_live"].strip(),
        pacing_from=start_day.isoformat(),
        feedback=feedback,
        config=cfg,
        matrix=matrix,
        daily=daily,
        todo=todo,
        checklist_text=checklist
    )

    return {
        "feedback": feedback,
        "matrix": matrix,
        "daily": daily,
        "todo": todo,
        "checklist": checklist,
    }

def build_checklist_text(cfg: dict) -> str:
    # Testo volutamente “operativo” e corto (copiabile)
    airbnb_weekly = float(cfg.get("airbnb_weekly_pct", 5))
    booking_nonref = float(cfg.get("booking_nonref_pct", 5))
    booking_weekly = float(cfg.get("booking_weekly_pct", 5))

    lines = []
    lines.append("SITO (Direct)")
    lines.append("- Regola: Direct sempre -15% vs OTA effective")
    lines.append("- Usa matrix_daily.csv colonna 'direct_2p' per prezzi giorno-per-giorno (include weekend + eventi).")
    lines.append("")

    lines.append("AIRBNB")
    lines.append("- Smart Pricing: OFF")
    lines.append("- Calendario: usa matrix_daily.csv colonna 'airbnb_2p' (2 ospiti)")
    lines.append(f"- Weekly discount: {airbnb_weekly:.0f}%")
    lines.append("- Extra guests: 3–5: +15€/ospite | 6–7: +10€/ospite (2→7 = +65€)")
    if bool(cfg.get("face20", True)):
        lines.append("- Promo “-20% visibile”: ON (NO stacking)")
        if not bool(cfg.get("airbnb_has20", True)):
            lines.append("  * TODO: attivare -20% su Airbnb (per evitare stacking).")
    lines.append("")

    lines.append("BOOKING")
    lines.append("- Rates & Availability: usa matrix_daily.csv colonna 'booking_2p'")
    lines.append(f"- Non-refundable: {booking_nonref:.0f}% più economico dello Standard")
    lines.append(f"- Weekly: {booking_weekly:.0f}% più economico dello Standard")
    lines.append("- Pricing per ospite: 3–5: +15€/ospite | 6–7: +10€/ospite (2→7 = +65€)")
    if bool(cfg.get("face20", True)) and not bool(cfg.get("booking_has20", True)):
        lines.append("- TODO: attivare -20% su Booking (per evitare stacking).")

    return "\n".join(lines)

class SimpleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x740")

        cfg = load_gui_config()

        # ESSENZIALI
        self.csv_url = tk.StringVar(value=cfg.get("csv_url", ""))
        self.go_live = tk.StringVar(value=cfg.get("go_live", "2026-02-10"))
        self.direct_base = tk.StringVar(value=str(cfg.get("direct_base", 60.0)))
        self.auto_adjust = tk.BooleanVar(value=cfg.get("auto_adjust", True))

        self.face20 = tk.BooleanVar(value=cfg.get("face20", True))
        self.airbnb_has20 = tk.BooleanVar(value=cfg.get("airbnb_has20", True))
        self.booking_has20 = tk.BooleanVar(value=cfg.get("booking_has20", True))
        self.daily_horizon = tk.StringVar(value=str(cfg.get("daily_horizon", 90)))

        # AVANZATE (restano, ma non “disturbano”)
        self.launch_floor = tk.StringVar(value=str(cfg.get("launch_floor", 60.0)))
        self.launch_days = tk.StringVar(value=str(cfg.get("launch_days", 21)))
        self.launch_bookings = tk.StringVar(value=str(cfg.get("launch_bookings", 3)))
        self.airbnb_weekly = tk.StringVar(value=str(cfg.get("airbnb_weekly_pct", 5)))
        self.booking_nonref = tk.StringVar(value=str(cfg.get("booking_nonref_pct", 5)))
        self.booking_weekly = tk.StringVar(value=str(cfg.get("booking_weekly_pct", 5)))

        self.state = None  # risultati
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="CSV URL (Google Sheet pubblicato in CSV):").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.csv_url, width=110).grid(row=1, column=0, columnspan=6, sticky="we")

        row2 = ttk.Frame(self)
        row2.pack(fill="x", **pad)

        ttk.Label(row2, text="Go-live:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row2, textvariable=self.go_live, width=14).grid(row=0, column=1, sticky="w")

        ttk.Label(row2, text="Direct base 2p feriale €:").grid(row=0, column=2, sticky="w", padx=(18,0))
        ttk.Entry(row2, textvariable=self.direct_base, width=8).grid(row=0, column=3, sticky="w")

        ttk.Checkbutton(row2, text="Auto-adjust (pacing)", variable=self.auto_adjust).grid(row=0, column=4, sticky="w", padx=(18,0))

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", **pad)

        ttk.Checkbutton(runbar, text="Promo faccia “-20%” ON", variable=self.face20).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(runbar, text="Airbnb: -20% già attiva", variable=self.airbnb_has20).grid(row=0, column=1, sticky="w", padx=(12,0))
        ttk.Checkbutton(runbar, text="Booking: -20% già attiva", variable=self.booking_has20).grid(row=0, column=2, sticky="w", padx=(12,0))

        ttk.Label(runbar, text="Orizzonte (giorni):").grid(row=0, column=3, sticky="e", padx=(20,0))
        ttk.Entry(runbar, textvariable=self.daily_horizon, width=6).grid(row=0, column=4, sticky="w")

        ttk.Button(runbar, text="GENERA PIANO", command=self.on_run).grid(row=0, column=5, sticky="e", padx=(20,0))

        # Avanzate in un box “richiudibile”
        self.adv_open = tk.BooleanVar(value=False)
        adv_toggle = ttk.Checkbutton(self, text="Mostra Avanzate", variable=self.adv_open, command=self._toggle_adv)
        adv_toggle.pack(anchor="w", padx=14, pady=(0,6))

        self.adv = ttk.LabelFrame(self, text="Avanzate (di solito non tocchi nulla)")
        # di default nascosta

        ttk.Label(self.adv, text="Launch floor €:").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        ttk.Entry(self.adv, textvariable=self.launch_floor, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(self.adv, text="Launch giorni:").grid(row=0, column=2, sticky="w", padx=10)
        ttk.Entry(self.adv, textvariable=self.launch_days, width=6).grid(row=0, column=3, sticky="w")

        ttk.Label(self.adv, text="Launch bookings soglia:").grid(row=0, column=4, sticky="w", padx=10)
        ttk.Entry(self.adv, textvariable=self.launch_bookings, width=6).grid(row=0, column=5, sticky="w")

        ttk.Label(self.adv, text="Airbnb weekly %:").grid(row=1, column=0, sticky="w", padx=10, pady=6)
        ttk.Entry(self.adv, textvariable=self.airbnb_weekly, width=6).grid(row=1, column=1, sticky="w")

        ttk.Label(self.adv, text="Booking non-ref %:").grid(row=1, column=2, sticky="w", padx=10)
        ttk.Entry(self.adv, textvariable=self.booking_nonref, width=6).grid(row=1, column=3, sticky="w")

        ttk.Label(self.adv, text="Booking weekly %:").grid(row=1, column=4, sticky="w", padx=10)
        ttk.Entry(self.adv, textvariable=self.booking_weekly, width=6).grid(row=1, column=5, sticky="w")

        # Output area
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_summary = ttk.Frame(self.notebook)
        self.tab_airbnb = ttk.Frame(self.notebook)
        self.tab_booking = ttk.Frame(self.notebook)
        self.tab_site = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_summary, text="Sommario")
        self.notebook.add(self.tab_airbnb, text="Airbnb")
        self.notebook.add(self.tab_booking, text="Booking")
        self.notebook.add(self.tab_site, text="Sito")

        self.summary_lbl = ttk.Label(self.tab_summary, text="Premi “GENERA PIANO”.", justify="left")
        self.summary_lbl.pack(anchor="nw", padx=10, pady=10)

        self.checklist_txt = tk.Text(self.tab_summary, height=18, wrap="word")
        self.checklist_txt.pack(fill="both", expand=True, padx=10, pady=(0,10))

        btns = ttk.Frame(self.tab_summary)
        btns.pack(fill="x", padx=10, pady=(0,10))
        ttk.Button(btns, text="Copia checklist", command=self.copy_checklist).pack(side="left")

        # Tables per canale
        self.airbnb_tree = self._make_tree(self.tab_airbnb, "airbnb_2p", "airbnb_7p")
        self.booking_tree = self._make_tree(self.tab_booking, "booking_2p", "booking_7p")
        self.site_tree = self._make_tree(self.tab_site, "direct_2p", "direct_7p")

        self._set_channel_intro(self.tab_airbnb, "AIRBNB — usa colonna 'airbnb_2p' (2 ospiti) + extra guest fees.")
        self._set_channel_intro(self.tab_booking, "BOOKING — usa colonna 'booking_2p' (Standard) + sconti Non-ref/Weekly.")
        self._set_channel_intro(self.tab_site, "SITO — usa colonna 'direct_2p' (Direct = -15% vs OTA effective).")

    def _toggle_adv(self):
        if self.adv_open.get():
            self.adv.pack(fill="x", padx=10, pady=(0,10))
        else:
            self.adv.pack_forget()

    def _set_channel_intro(self, tab, text):
        lbl = ttk.Label(tab, text=text, justify="left")
        lbl.pack(anchor="nw", padx=10, pady=(10,4))

    def _make_tree(self, parent, col_2p, col_7p):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        cols = ("date", "dow", "peak", col_2p, col_7p, "label")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120 if c in (col_2p, col_7p) else 140, anchor="w")
        tree.column("dow", width=70)
        tree.column("peak", width=70)
        tree.column("label", width=360)
        tree.pack(fill="both", expand=True)

        return tree

    def _cfg_from_ui(self) -> dict:
        try:
            # validazioni minime
            if not self.csv_url.get().strip().startswith("http"):
                raise ValueError("CSV URL non valido (deve iniziare con http).")
            parse_yyyy_mm_dd(self.go_live.get().strip())
            float(self.direct_base.get().strip().replace(",", "."))
        except Exception as e:
            raise ValueError(str(e))

        cfg = {
            "csv_url": self.csv_url.get().strip(),
            "go_live": self.go_live.get().strip(),
            "direct_base": float(self.direct_base.get().strip().replace(",", ".")),
            "auto_adjust": bool(self.auto_adjust.get()),
            "launch_floor": float(self.launch_floor.get().strip().replace(",", ".")),
            "launch_days": int(float(self.launch_days.get().strip())),
            "launch_bookings": int(float(self.launch_bookings.get().strip())),
            "face20": bool(self.face20.get()),
            "airbnb_has20": bool(self.airbnb_has20.get()),
            "booking_has20": bool(self.booking_has20.get()),
            "airbnb_weekly_pct": float(self.airbnb_weekly.get() or 5),
            "booking_nonref_pct": float(self.booking_nonref.get() or 5),
            "booking_weekly_pct": float(self.booking_weekly.get() or 5),
            "daily_horizon": int(float(self.daily_horizon.get() or 90)),
        }
        return cfg

    def on_run(self):
        try:
            cfg = self._cfg_from_ui()
        except Exception as e:
            messagebox.showerror("Errore input", str(e))
            return

        try:
            save_gui_config(cfg)
        except Exception:
            pass

        try:
            self.state = run_engine(cfg)
        except Exception as e:
            messagebox.showerror("Errore", str(e))
            return

        fb = self.state["feedback"]
        delta = fb["delta_vs_benchmark_pct"]
        delta_txt = f"{delta:+.0%}" if delta is not None else "n/d"

        summary = (
            f"Pacing da: {fb['pacing_from']}\n"
            f"Occ 14g: {fb['occ14_booked']}/{fb['occ14_total']} = {fb['occ14_pct']:.0%} (target {fb['target14']:.0%})\n"
            f"Occ 30g: {fb['occ30_booked']}/{fb['occ30_total']} = {fb['occ30_pct']:.0%} (target {fb['target30']:.0%})\n"
            f"Prenotazioni future (da go-live): {fb['future_bookings']}\n\n"
            f"Direct base suggerito: {fb['direct_base_suggested']}€ (tu: {fb['direct_base_input']}€)\n"
            f"Motivo: {fb['why']}\n"
            f"Launch: {'ON' if fb['in_launch'] else 'OFF'} (floor {fb['launch_floor']}€)\n"
            f"Delta vs benchmark: {delta_txt}\n\n"
            f"Output creati: recommendations.txt, matrix_monthly.csv, matrix_daily.csv, settings_recap.json"
        )
        self.summary_lbl.config(text=summary)

        self.checklist_txt.delete("1.0", "end")
        self.checklist_txt.insert("1.0", self.state["checklist"])

        self._fill_tables(self.state["daily"])

        self.notebook.select(self.tab_summary)

    def _fill_tables(self, daily_rows):
        # pulisci
        for t in (self.airbnb_tree, self.booking_tree, self.site_tree):
            for iid in t.get_children():
                t.delete(iid)

        # mostra 14 giorni (semplice e utile)
        for r in daily_rows[:14]:
            common = (r["date"], r["weekday"], f"x{r['peak_mult']:.2f}", r["peak_label"] or "")
            self.airbnb_tree.insert("", "end", values=(common[0], common[1], common[2], r["airbnb_2p"], r["airbnb_7p"], common[3]))
            self.booking_tree.insert("", "end", values=(common[0], common[1], common[2], r["booking_2p"], r["booking_7p"], common[3]))
            self.site_tree.insert("", "end", values=(common[0], common[1], common[2], r["direct_2p"], r["direct_7p"], common[3]))

    def copy_checklist(self):
        if not self.state:
            return
        self.clipboard_clear()
        self.clipboard_append(self.state["checklist"])
        messagebox.showinfo("Copiato", "Checklist copiata negli appunti.")

if __name__ == "__main__":
    SimpleApp().mainloop()
