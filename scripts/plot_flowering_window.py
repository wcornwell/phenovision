#!/usr/bin/env python
"""
Plot the 'flowering window' from the local flower/fruit predictions.

x-axis : calendar date (day-of-year, years merged onto a single Jan-Dec axis)
y-axis : species (limited to those with >MIN_OBS observations), labels coloured
         by genus
point  : one observation, coloured by whether flowering was detected (yes/no)

Species are ordered by their median *flowering* date so the plot reads as a
phenological sequence (early -> late bloomers). All taxa in this dataset are
Rutaceae, so genus is the informative taxonomic grouping for the labels.

Usage:
  .venv/bin/python scripts/plot_flowering_window.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PRED_CSV = "output/flowerfruit_local_test/predictions.csv"
OUT_PNG = "output/flowerfruit_local_test/flowering_window.png"
MIN_OBS = 50  # keep species with strictly more than this many observations

YES_COLOR = "#d1495b"   # flowering
NO_COLOR = "#cdd4d9"    # not flowering
MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    df = pd.read_csv(PRED_CSV)
    df = df[df["observed_on"].notna() & (df["observed_on"].astype(str).str.strip() != "")]

    # calendar date -> day of year (merge years onto one Jan-Dec axis)
    dt = pd.to_datetime(df["observed_on"], errors="coerce")
    df = df[dt.notna()].copy()
    df["doy"] = dt[dt.notna()].dt.dayofyear

    # flower_detected is written as the strings "True"/"False"
    df["flowering"] = df["flower_detected"].astype(str).str.strip().str.lower() == "true"

    # limit to well-sampled species
    counts = df["taxon_name"].value_counts()
    keep = counts[counts > MIN_OBS].index
    df = df[df["taxon_name"].isin(keep)].copy()

    # genus = first token of the binomial (all taxa here are Rutaceae)
    df["genus"] = df["taxon_name"].str.split().str[0]

    # Order species by median flowering day-of-year (early bloomers first).
    # Fall back to overall median for any species with no flowering detections.
    flowering_med = df[df["flowering"]].groupby("taxon_name")["doy"].median()
    overall_med = df.groupby("taxon_name")["doy"].median()
    order_key = overall_med.copy()
    order_key.loc[flowering_med.index] = flowering_med
    species_order = order_key.sort_values().index.tolist()
    y_of = {sp: i for i, sp in enumerate(species_order)}
    df["y"] = df["taxon_name"].map(y_of)

    n_species = len(species_order)

    # genus -> colour
    genera = sorted(df["genus"].unique())
    cmap = plt.get_cmap("tab10")
    genus_color = {g: cmap(i % 10) for i, g in enumerate(genera)}
    species_genus = df.drop_duplicates("taxon_name").set_index("taxon_name")["genus"]

    rng = np.random.default_rng(0)
    jitter = (rng.random(len(df)) - 0.5) * 0.55
    yj = df["y"].to_numpy() + jitter

    fig_h = max(5.0, n_species * 0.34)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    no = ~df["flowering"].to_numpy()
    yes = df["flowering"].to_numpy()
    ax.scatter(df["doy"].to_numpy()[no], yj[no], s=14, c=NO_COLOR,
               alpha=0.55, linewidths=0, zorder=2)
    ax.scatter(df["doy"].to_numpy()[yes], yj[yes], s=20, c=YES_COLOR,
               alpha=0.80, linewidths=0, zorder=3)

    ax.set_yticks(range(n_species))
    ax.set_yticklabels(species_order, fontsize=9, fontstyle="italic")
    for tick, sp in zip(ax.get_yticklabels(), species_order):
        tick.set_color(genus_color[species_genus[sp]])
    ax.set_ylim(-1, n_species)
    ax.invert_yaxis()  # earliest bloomers at top

    ax.set_xticks(MONTH_STARTS)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlim(1, 366)
    ax.set_xlabel("Calendar date (years merged)")
    ax.set_title(
        f"Flowering window — {n_species} Rutaceae species with >{MIN_OBS} obs, "
        f"ordered by median flowering date",
        fontsize=11,
    )
    for x in MONTH_STARTS:
        ax.axvline(x, color="0.92", lw=0.6, zorder=0)
    ax.grid(axis="y", color="0.95", lw=0.5, zorder=0)

    flower_legend = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=YES_COLOR,
                   markersize=8, label="flowering"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=NO_COLOR,
                   markersize=8, label="not flowering"),
        ],
        loc="upper right", framealpha=0.9, fontsize=9, title="detection",
    )
    ax.add_artist(flower_legend)
    ax.legend(
        handles=[Line2D([0], [0], marker="s", color="none",
                        markerfacecolor=genus_color[g], markersize=9, label=g)
                 for g in genera],
        loc="lower right", framealpha=0.9, fontsize=8, title="genus",
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}  ({n_species} species, {len(df)} obs, {len(genera)} genera)")


if __name__ == "__main__":
    main()
