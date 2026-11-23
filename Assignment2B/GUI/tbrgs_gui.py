"""
tbrgs_gui.py - Traffic-based Route Guidance System (TBRGS) GUI, COS30019 Assignment 2B

One window that runs Tasks 1-4 together:
  Route finder tab
      Inputs : origin, destination, date, time, prediction model (LSTM / GRU / XGBoost regression),
               search method from Assignment 2A (DFS, BFS, GBFS, AS, CUS1, CUS2), number of routes.
      Output : up to 5 routes with their travel times, the selected route drawn on a map of the
               40 SCATS sites (roads coloured by predicted speed), and a road-by-road table.
  Model evaluation tab
      The comparison table (MAE, RMSE, NRMSE, MAPE, R2) and charts from evaluate_models.py,
      with a button to run the evaluation again.

Defaults come from config.json; "Save as default" writes the current choices back to it.

Run (from the Assignment2B folder, after training the models):
    python GUI/tbrgs_gui.py
Needs: python -m pip install -r requirements.txt   (includes customtkinter)
"""
import json
import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

try:
    import customtkinter as ctk
except ImportError:
    sys.exit("This GUI needs CustomTkinter. Install it with:  python -m pip install customtkinter")

import matplotlib
matplotlib.use("TkAgg")
import pandas as pd                                                       # noqa: E402
from matplotlib import cm                                                 # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk   # noqa: E402
from matplotlib.colors import Normalize                                   # noqa: E402
from matplotlib.figure import Figure                                      # noqa: E402
from PIL import Image                                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # the Assignment2B folder (GUI/..)
sys.path.insert(0, os.path.join(ROOT, "route_finding"))
from route_finder import ALGORITHM_NAMES, ALGORITHMS, CONFIG_FILE, RouteFinder, load_config   # noqa: E402
from travel_time import CAPACITY_SPEED, SPEED_LIMIT                       # noqa: E402

MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "xgboost": "XGBoost"}
MODEL_KEYS = {v: k for k, v in MODEL_LABELS.items()}
EVAL_DIR = os.path.join(ROOT, "results", "evaluation")
EVAL_SCRIPT = os.path.join(ROOT, "Machine_Learning_Algorithm", "evaluate_models.py")
EVAL_CHARTS = {"Metrics": "metrics_comparison.png",
               "Minutes ahead": "error_by_horizon.png",
               "By hour": "error_by_hour.png",
               "Per site": "per_site_mae.png",
               "Actual vs predicted": "actual_vs_predicted.png",
               "Scatter": "scatter_actual_vs_predicted.png"}
ROUTE_COLOUR = "#1f6feb"
ALT_COLOUR = "#6e7781"


class TBRGSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title("TBRGS - Traffic-based Route Guidance System (Boroondara)")
        self.geometry("1420x880")
        self.minsize(1150, 720)

        self.config_data = load_config()
        self.finder = RouteFinder(self.config_data)
        self.site_names = self.finder.sites()                           # {site: "ROAD / ROAD"}
        self.site_options = [f"{s}  {name}" for s, name in sorted(self.site_names.items())]
        first, last = self.finder.time_range()
        self.first_time, self.last_time = first, last
        days = pd.date_range(first.normalize(), last.normalize(), freq="D")
        self.date_options = {d.strftime("%a %d %b %Y"): d for d in days}
        self.time_options = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]

        self.jobs = queue.Queue()            # results coming back from worker threads
        self.result = None
        self.speeds = None

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        self.route_tab = self.tabs.add("Route finder")
        self.eval_tab = self.tabs.add("Model evaluation")
        self._build_route_tab()
        self._build_eval_tab()
        self._apply_defaults()
        self.draw_map()
        self.load_evaluation()
        self.after(100, self._poll_jobs)

    # ================================================================ route finder tab
    def _build_route_tab(self):
        tab = self.route_tab
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # ---------------- left: inputs
        left = ctk.CTkScrollableFrame(tab, width=330, label_text="Plan a trip")
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        self._section(left, "Origin")
        self.origin_box = ctk.CTkComboBox(left, values=self.site_options, width=300)
        self.origin_box.pack(anchor="w", padx=8)
        ctk.CTkButton(left, text="⇅  Swap origin and destination", width=300, height=26,
                      fg_color="transparent", border_width=1, text_color=("gray20", "gray85"),
                      command=self.swap_sites).pack(anchor="w", padx=8, pady=6)
        self._section(left, "Destination")
        self.dest_box = ctk.CTkComboBox(left, values=self.site_options, width=300)
        self.dest_box.pack(anchor="w", padx=8)

        self._section(left, "Departure date and time")
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(anchor="w", padx=8)
        self.date_box = ctk.CTkOptionMenu(row, values=list(self.date_options), width=180)
        self.date_box.pack(side="left")
        self.time_box = ctk.CTkComboBox(row, values=self.time_options, width=110)
        self.time_box.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(left, text=f"Data covers {self.first_time:%d %b %H:%M} - {self.last_time:%d %b %Y %H:%M}",
                     font=ctk.CTkFont(size=11), text_color="gray50").pack(anchor="w", padx=8)

        self._section(left, "Traffic prediction model")
        self.model_seg = ctk.CTkSegmentedButton(left, values=list(MODEL_KEYS), width=300)
        self.model_seg.pack(anchor="w", padx=8)

        self._section(left, "Search method (Assignment 2A)")
        self.algo_var = tk.StringVar()
        grid = ctk.CTkFrame(left, fg_color="transparent")
        grid.pack(anchor="w", padx=8)
        for i, key in enumerate(ALGORITHMS):
            ctk.CTkRadioButton(grid, text=f"{key}  {ALGORITHM_NAMES[key].split(' (')[0]}",
                               variable=self.algo_var, value=key,
                               font=ctk.CTkFont(size=12)).grid(row=i, column=0, sticky="w", pady=3)

        self._section(left, "Number of routes")
        self.routes_seg = ctk.CTkSegmentedButton(left, values=["1", "2", "3", "4", "5"], width=300)
        self.routes_seg.pack(anchor="w", padx=8)

        self.find_btn = ctk.CTkButton(left, text="Find routes", height=40, width=300,
                                      font=ctk.CTkFont(size=15, weight="bold"), command=self.find_routes)
        self.find_btn.pack(anchor="w", padx=8, pady=(18, 6))
        self.progress = ctk.CTkProgressBar(left, mode="indeterminate", width=300)
        self.status = ctk.CTkLabel(left, text="", wraplength=300, justify="left", font=ctk.CTkFont(size=12))
        self.status.pack(anchor="w", padx=8, pady=(2, 8))

        bottom = ctk.CTkFrame(left, fg_color="transparent")
        bottom.pack(anchor="w", padx=8, pady=(6, 4))
        ctk.CTkButton(bottom, text="Save as default", width=140, height=28, fg_color="gray55",
                      hover_color="gray45", command=self.save_defaults).pack(side="left")
        self.dark_switch = ctk.CTkSwitch(bottom, text="Dark mode", command=self.toggle_dark)
        self.dark_switch.pack(side="left", padx=(16, 0))

        # ---------------- right: summary, map, routes, road table
        right = ctk.CTkFrame(tab)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=3)
        right.grid_rowconfigure(2, weight=1)

        summary = ctk.CTkFrame(right, fg_color="transparent")
        summary.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))
        self.time_label = ctk.CTkLabel(summary, text="-- min", font=ctk.CTkFont(size=30, weight="bold"))
        self.time_label.pack(side="left")
        self.summary_label = ctk.CTkLabel(summary, text="Choose a trip and press Find routes.",
                                          justify="left", font=ctk.CTkFont(size=13))
        self.summary_label.pack(side="left", padx=16)

        map_frame = tk.Frame(right, bg="white")
        map_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        self.fig = Figure(figsize=(8, 5.5), dpi=100)
        self.ax = self.fig.add_axes([0.05, 0.07, 0.83, 0.86])
        self.cax = self.fig.add_axes([0.9, 0.2, 0.018, 0.6])
        self.canvas = FigureCanvasTkAgg(self.fig, master=map_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, map_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        lower = ctk.CTkFrame(right, fg_color="transparent")
        lower.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 10))
        lower.grid_columnconfigure(1, weight=1)
        lower.grid_rowconfigure(0, weight=1)
        self.routes_frame = ctk.CTkScrollableFrame(lower, width=430, label_text="Routes (click to show)")
        self.routes_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.route_var = tk.IntVar(value=0)

        table_frame = ctk.CTkFrame(lower)
        table_frame.grid(row=0, column=1, sticky="nsew")
        cols = ("from", "to", "road", "km", "speed", "time")
        heads = ("From", "To", "Road", "Distance (km)", "Speed (km/h)", "Time (s)")
        self.legs_table = ttk.Treeview(table_frame, columns=cols, show="headings", height=6)
        for c, h, w in zip(cols, heads, (60, 60, 150, 100, 100, 80)):
            self.legs_table.heading(c, text=h)
            self.legs_table.column(c, width=w, anchor="w" if c == "road" else "center")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.legs_table.yview)
        self.legs_table.configure(yscrollcommand=scroll.set)
        self.legs_table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._style_tables()

    def _section(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=8, pady=(14, 4))

    def _apply_defaults(self):
        c = self.config_data
        self.origin_box.set(self._site_option(c.get("default_origin", "2000")))
        self.dest_box.set(self._site_option(c.get("default_destination", "3002")))
        when = pd.Timestamp(c.get("default_datetime", "2006-10-30 08:00")).floor("15min")
        self.date_box.set(when.strftime("%a %d %b %Y"))
        self.time_box.set(when.strftime("%H:%M"))
        self.model_seg.set(MODEL_LABELS.get(c.get("default_model", "xgboost"), "XGBoost"))
        self.algo_var.set(c.get("default_algorithm", "AS").upper())
        self.routes_seg.set(str(c.get("max_routes", 5)))

    def _site_option(self, site):
        site = str(site).zfill(4)
        return f"{site}  {self.site_names.get(site, '')}"

    def _inputs(self):
        origin = self.origin_box.get().split()[0] if self.origin_box.get().strip() else ""
        dest = self.dest_box.get().split()[0] if self.dest_box.get().strip() else ""
        day = self.date_options.get(self.date_box.get())
        when = f"{day:%Y-%m-%d} {self.time_box.get().strip()}" if day is not None else self.time_box.get()
        return {"origin": origin, "destination": dest, "when": when,
                "model": MODEL_KEYS[self.model_seg.get()], "algorithm": self.algo_var.get(),
                "max_routes": int(self.routes_seg.get())}

    def swap_sites(self):
        o, d = self.origin_box.get(), self.dest_box.get()
        self.origin_box.set(d)
        self.dest_box.set(o)

    # ---------------------------------------------------------------- running a search
    def find_routes(self):
        args = self._inputs()
        self.find_btn.configure(state="disabled")
        self.progress.pack(anchor="w", padx=8, after=self.find_btn)
        self.progress.start()
        first_use = args["model"] not in self.finder._estimators
        self._set_status("Loading the model and predicting traffic..." if first_use
                         else "Predicting traffic and searching...", "gray40")

        def work():
            try:
                result = self.finder.find_routes(args["origin"], args["destination"], args["when"],
                                                 model=args["model"], algorithm=args["algorithm"],
                                                 max_routes=args["max_routes"])
                speeds = self.finder.estimator(args["model"]).edge_times(result["time"])
                self.jobs.put(("routes", (result, speeds)))
            except ValueError as error:
                self.jobs.put(("error", str(error)))
            except Exception as error:                       # unexpected problem: show it, keep the GUI open
                self.jobs.put(("error", f"Something went wrong: {error}"))
        threading.Thread(target=work, daemon=True).start()

    def _poll_jobs(self):
        try:
            while True:
                kind, payload = self.jobs.get_nowait()
                if kind == "routes":
                    self._done()
                    self.show_result(*payload)
                elif kind == "error":
                    self._done()
                    self._set_status(payload, "#c0392b")
                elif kind == "eval_done":
                    self.eval_btn.configure(state="normal")
                    self.eval_progress.stop()
                    self.eval_progress.pack_forget()
                    self.eval_status.configure(text=payload)
                    self.load_evaluation()
        except queue.Empty:
            pass
        self.after(100, self._poll_jobs)

    def _done(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.find_btn.configure(state="normal")

    def _set_status(self, text, colour):
        self.status.configure(text=text, text_color=colour)

    def show_result(self, result, speeds):
        self.result, self.speeds = result, speeds
        for child in self.routes_frame.winfo_children():
            child.destroy()
        routes = result["routes"]
        if not routes:
            self._set_status("No route found between these sites.", "#c0392b")
            return
        for i, r in enumerate(routes):
            tags = []
            if i == 0:
                tags.append(f"found by {result['algorithm']}")
            if i == result["fastest_index"]:
                tags.append("fastest")
            path = r["path"]
            short = " → ".join(path) if len(path) <= 5 else f"{' → '.join(path[:3])} → … → {path[-1]}"
            text = f"Route {i + 1}:  {r['travel_time_min']} min, {r['distance_km']:.2f} km" + \
                   (f"  ({', '.join(tags)})" if tags else "") + f"\n{short}"
            ctk.CTkRadioButton(self.routes_frame, text=text, variable=self.route_var, value=i,
                               command=self.show_selected).pack(anchor="w", pady=4)
        self.route_var.set(result["fastest_index"])
        self.show_selected()
        self._set_status(f"{result['algorithm_name']} created {result['nodes_created']} nodes "
                         f"({result['search_ms']} ms). Route 1 is its own answer; routes 2+ are "
                         f"alternatives found by the same method.", "gray30")

    def show_selected(self):
        result, i = self.result, self.route_var.get()
        route = result["routes"][i]
        self.time_label.configure(text=f"{route['travel_time_min']} min")
        fastest = " - the fastest found" if i == result["fastest_index"] else ""
        when = pd.Timestamp(result["time"])
        self.summary_label.configure(
            text=f"Route {i + 1}{fastest}\n"
                 f"{result['origin']} {self.site_names[result['origin']]}  →  "
                 f"{result['destination']} {self.site_names[result['destination']]}\n"
                 f"{route['distance_km']:.2f} km, {route['intersections']} intersections | "
                 f"leaving {when:%a %d %b %Y %H:%M} | {MODEL_LABELS[result['model']]} | {result['algorithm']}")
        self.legs_table.delete(*self.legs_table.get_children())
        for leg in route["legs"]:
            self.legs_table.insert("", "end", values=(leg["from"], leg["to"], leg["road"],
                                                      f"{leg['distance_km']:.2f}", f"{leg['speed_kmh']:.1f}",
                                                      f"{leg['travel_time_s']:.0f}"))
        self.draw_map()

    # ---------------------------------------------------------------- map
    def draw_map(self):
        est = self.finder.estimator(self.config_data.get("default_model", "xgboost"))
        sites, edges = est.sites, est.edges
        ax, cax = self.ax, self.cax
        ax.clear()
        cax.clear()
        cmap = cm.get_cmap("RdYlGn") if hasattr(cm, "get_cmap") else matplotlib.colormaps["RdYlGn"]
        norm = Normalize(vmin=CAPACITY_SPEED, vmax=SPEED_LIMIT)

        slowest = {}
        if self.speeds is not None:
            for _, r in self.speeds.iterrows():
                key = tuple(sorted((r["from"], r["to"])))
                slowest[key] = min(slowest.get(key, SPEED_LIMIT), r["speed_kmh"])
        for _, e in edges.iterrows():
            a, b = e["from_site"], e["to_site"]
            speed = slowest.get(tuple(sorted((a, b))))
            colour = cmap(norm(speed)) if speed is not None else "#c9ced6"
            ax.plot([sites.at[a, "long"], sites.at[b, "long"]], [sites.at[a, "lat"], sites.at[b, "lat"]],
                    color=colour, lw=2.5, alpha=0.75, zorder=1, solid_capstyle="round")

        title = "SCATS sites in Boroondara (roads coloured by predicted speed after a search)"
        if self.result and self.result["routes"]:
            chosen = self.route_var.get()
            for i, r in enumerate(self.result["routes"]):
                if i != chosen:
                    self._draw_path(r["path"], sites, ALT_COLOUR, 2, "--", 2)
            self._draw_path(self.result["routes"][chosen]["path"], sites, ROUTE_COLOUR, 5, "-", 3)
            o, d = self.result["origin"], self.result["destination"]
            ax.scatter(sites.at[o, "long"], sites.at[o, "lat"], s=260, c="#2da44e", edgecolors="black",
                       zorder=5, label="Origin")
            ax.scatter(sites.at[d, "long"], sites.at[d, "lat"], s=260, c="#cf222e", edgecolors="black",
                       zorder=5, label="Destination")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
            title = (f"Route {chosen + 1} of {len(self.result['routes'])}: {o} → {d}   "
                     f"({MODEL_LABELS[self.result['model']]}, {self.result['algorithm']}, "
                     f"{pd.Timestamp(self.result['time']):%d %b %H:%M})")

        ax.scatter(sites["long"], sites["lat"], s=40, c="white", edgecolors="#333333", zorder=4)
        for s in sites.index:
            ax.annotate(s, (sites.at[s, "long"], sites.at[s, "lat"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, zorder=6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect(1 / math.cos(math.radians(sites["lat"].mean())), adjustable="datalim")
        bar = self.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        bar.set_label("Predicted speed (km/h)", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        self.canvas.draw_idle()

    def _draw_path(self, path, sites, colour, width, style, z):
        xs = [sites.at[s, "long"] for s in path]
        ys = [sites.at[s, "lat"] for s in path]
        self.ax.plot(xs, ys, style, color=colour, lw=width, zorder=z, solid_capstyle="round")

    # ================================================================ evaluation tab
    def _build_eval_tab(self):
        tab = self.eval_tab
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        ctk.CTkLabel(top, text="Model comparison on the test days (27-31 October 2006, all 40 sites)",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        self.eval_btn = ctk.CTkButton(top, text="Run evaluation again", width=170, command=self.run_evaluation)
        self.eval_btn.pack(side="right")
        self.eval_progress = ctk.CTkProgressBar(top, mode="indeterminate", width=160)
        self.eval_status = ctk.CTkLabel(tab, text="", justify="left", text_color="gray40")
        self.eval_status.grid(row=1, column=0, sticky="w")

        table_frame = ctk.CTkFrame(tab)
        table_frame.grid(row=2, column=0, sticky="ew", pady=(4, 8))
        cols = ("model", "MAE", "RMSE", "NRMSE", "MAPE_%", "R2", "train_seconds")
        heads = ("Model", "MAE", "RMSE", "NRMSE", "MAPE %", "R²", "Training time (s)")
        self.eval_table = ttk.Treeview(table_frame, columns=cols, show="headings", height=5)
        for c, h in zip(cols, heads):
            self.eval_table.heading(c, text=h)
            self.eval_table.column(c, width=200 if c == "model" else 110, anchor="w" if c == "model" else "center")
        self.eval_table.pack(fill="x", expand=True)

        chart_area = ctk.CTkFrame(tab)
        chart_area.grid(row=3, column=0, sticky="nsew")
        self.chart_seg = ctk.CTkSegmentedButton(chart_area, values=list(EVAL_CHARTS), command=self.show_chart)
        self.chart_seg.pack(pady=8)
        self.chart_seg.set("Metrics")
        self.chart_label = ctk.CTkLabel(chart_area, text="")
        self.chart_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        ctk.CTkLabel(tab, text="MAE / RMSE: average and large-miss error in cars per 15 min (lower is better).  "
                               "NRMSE: RMSE as a share of average traffic.  MAPE: error in %.  "
                               "R²: share of the traffic pattern explained (1 = perfect).",
                     font=ctk.CTkFont(size=11), text_color="gray45", wraplength=1200,
                     justify="left").grid(row=4, column=0, sticky="w", pady=(4, 0))

    def load_evaluation(self):
        self.eval_table.delete(*self.eval_table.get_children())
        table_file = os.path.join(EVAL_DIR, "comparison_table.csv")
        if not os.path.exists(table_file):
            self.eval_status.configure(text="No evaluation results yet. Train the three models, then "
                                            "press 'Run evaluation again'.")
            self.chart_label.configure(image=None, text="No charts yet.")
            return
        table = pd.read_csv(table_file, index_col=0)
        for name, row in table.iterrows():
            train = row.get("train_seconds")
            self.eval_table.insert("", "end", values=(
                name, f"{row['MAE']:.2f}", f"{row['RMSE']:.2f}", f"{row['NRMSE']:.3f}",
                f"{row['MAPE_%']:.2f}", f"{row['R2']:.4f}", "-" if pd.isna(train) else f"{train:.1f}"))
        best = table.loc[[n for n in table.index if n in ("LSTM", "GRU", "XGBoost regression")], "MAE"].idxmin() \
            if any(n in table.index for n in ("LSTM", "GRU", "XGBoost regression")) else None
        if best and not self.eval_status.cget("text"):
            self.eval_status.configure(text=f"Lowest error: {best}. Baselines (Persistence, Historical "
                                            f"average) show what the models must beat.")
        self.show_chart(self.chart_seg.get())

    def show_chart(self, name):
        path = os.path.join(EVAL_DIR, EVAL_CHARTS[name])
        if not os.path.exists(path):
            self.chart_label.configure(image=None, text=f"{EVAL_CHARTS[name]} not found.")
            return
        image = Image.open(path)
        max_w = max(self.chart_label.winfo_width() - 20, 1100)       # fit the chart to the space available
        max_h = max(self.chart_label.winfo_height() - 20, 430)
        scale = min(max_w / image.width, max_h / image.height)
        size = (int(image.width * scale), int(image.height * scale))
        self.chart_image = ctk.CTkImage(light_image=image, dark_image=image, size=size)
        self.chart_label.configure(image=self.chart_image, text="")

    def run_evaluation(self):
        self.eval_btn.configure(state="disabled")
        self.eval_progress.pack(side="right", padx=10)
        self.eval_progress.start()
        self.eval_status.configure(text="Running evaluate_models.py (about 10-20 seconds)...")

        def work():
            done = subprocess.run([sys.executable, EVAL_SCRIPT], cwd=ROOT, capture_output=True, text=True)
            if done.returncode == 0:
                message = "Evaluation finished. Tables and charts updated."
            else:
                message = "Evaluation failed: " + (done.stderr.strip().splitlines() or ["unknown error"])[-1]
            self.jobs.put(("eval_done", message))
        threading.Thread(target=work, daemon=True).start()

    # ================================================================ settings
    def save_defaults(self):
        args = self._inputs()
        self.config_data.update({"default_origin": args["origin"], "default_destination": args["destination"],
                                 "default_model": args["model"], "default_algorithm": args["algorithm"],
                                 "default_datetime": args["when"], "max_routes": args["max_routes"]})
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, indent=4)
        self._set_status("Saved these choices as the defaults in config.json.", "#2da44e")

    def toggle_dark(self):
        ctk.set_appearance_mode("dark" if self.dark_switch.get() else "light")
        self._style_tables()

    def _style_tables(self):
        dark = ctk.get_appearance_mode() == "Dark"
        style = ttk.Style()
        style.theme_use("clam")
        bg, fg, head = ("#2b2b2b", "#e6e6e6", "#3a3a3a") if dark else ("#ffffff", "#1f1f1f", "#e8ecf1")
        style.configure("Treeview", background=bg, fieldbackground=bg, foreground=fg, rowheight=24,
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background=head, foreground=fg, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", ROUTE_COLOUR)], foreground=[("selected", "white")])


def main():
    app = TBRGSApp()
    app.mainloop()


if __name__ == "__main__":
    main()
