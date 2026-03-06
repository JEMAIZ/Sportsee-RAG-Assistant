# plot_tool.py
import json
import logging
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from langchain.tools import tool
from utils.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)
Path(f"{OUTPUTS_DIR}/graphs").mkdir(parents=True, exist_ok=True)


def _plot_bar(ax, data, x_key, y_key, title):
    labels = [str(row.get(x_key, "?")) for row in data]
    values = [float(row.get(y_key, 0)) for row in data]
    colors = plt.cm.RdYlGn([v / max(values) if max(values) > 0 else 0 for v in values])
    bars   = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel(y_key)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)


def _plot_line(ax, data, x_key, y_key, title):
    x_vals = [str(row.get(x_key, i)) for i, row in enumerate(data)]
    y_vals = [float(row.get(y_key, 0)) for row in data]
    ax.plot(x_vals, y_vals, marker="o", linewidth=2, markersize=6,
            color="#2196F3", markerfacecolor="#FF5722")
    ax.fill_between(range(len(x_vals)), y_vals, alpha=0.15, color="#2196F3")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(y_key)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def _plot_scatter(ax, data, x_key, y_key, title):
    x_vals = [float(row.get(x_key, 0)) for row in data]
    y_vals = [float(row.get(y_key, 0)) for row in data]
    labels = [str(row.get("player", "")) for row in data]
    ax.scatter(x_vals, y_vals, alpha=0.7, s=60, c="#673AB7")
    for x, y, lbl in zip(x_vals, y_vals, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, alpha=0.8)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(linestyle="--", alpha=0.4)


def _plot_pie(ax, data, x_key, y_key, title):
    labels = [str(row.get(x_key, "?")) for row in data]
    values = [abs(float(row.get(y_key, 0))) for row in data]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%", startangle=90, pctdistance=0.85
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)


def generate_chart(data, chart_type="bar", x_key="player",
                   y_key="pts", title="Statistiques NBA",
                   return_base64=True) -> str:
    if not data:
        raise ValueError("Aucune donnee fournie pour le graphique.")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    CHART_FUNCTIONS = {
        "bar":     _plot_bar,
        "line":    _plot_line,
        "scatter": _plot_scatter,
        "pie":     _plot_pie,
    }
    plot_fn = CHART_FUNCTIONS.get(chart_type, _plot_bar)
    plot_fn(ax, data, x_key, y_key, title)
    plt.tight_layout()

    if return_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{b64}"
    else:
        path = f"{OUTPUTS_DIR}/graphs/{title[:30].replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


@tool
def plot_tool(plot_request: str) -> str:
    """
    Genere un graphique a partir de donnees NBA structurees.
    L'input doit etre un JSON avec les champs :
      - data       : liste de dicts avec les donnees
      - chart_type : bar | line | scatter | pie
      - x_key      : colonne pour l'axe X (ex: player)
      - y_key      : colonne pour l'axe Y (ex: pts)
      - title      : titre du graphique

    Exemple :
    {
      "data": [{"player": "LeBron", "pts": 25.7}, {"player": "Curry", "pts": 26.4}],
      "chart_type": "bar",
      "x_key": "player",
      "y_key": "pts",
      "title": "Points par match"
    }
    """
    logger.info("Plot Tool appele")

    try:
        params = json.loads(plot_request)
    except json.JSONDecodeError as e:
        return f"Erreur de parsing JSON : {e}"

    for field in ["data", "x_key", "y_key", "title"]:
        if field not in params:
            return f"Champ requis manquant : {field}"

    try:
        result = generate_chart(
            data=params["data"],
            chart_type=params.get("chart_type", "bar"),
            x_key=params["x_key"],
            y_key=params["y_key"],
            title=params["title"],
            return_base64=True
        )
        return f"GRAPH_BASE64:{result}"
    except Exception as e:
        logger.error(f"Erreur graphique: {e}")
        return f"Erreur lors de la generation du graphique : {e}"