# plot_tool.py
import json
import logging
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from langchain.tools import tool
from utils.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)

GRAPHS_DIR = Path(f"{OUTPUTS_DIR}/graphs")
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def _plot_bar(ax, data, x_key, y_key, title):
    labels = [str(row.get(x_key, "?")) for row in data]
    values = [float(row.get(y_key, 0)) for row in data]
    max_v  = max(values) if values else 1
    colors = plt.cm.RdYlGn([v / max_v for v in values])
    bars   = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel(y_key)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + max_v * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=8
        )


def _plot_line(ax, data, x_key, y_key, title):
    x_vals = [str(row.get(x_key, i)) for i, row in enumerate(data)]
    y_vals = [float(row.get(y_key, 0)) for row in data]
    ax.plot(x_vals, y_vals, marker="o", linewidth=2, markersize=5,
            color="#2196F3", markerfacecolor="#FF5722")
    ax.fill_between(range(len(x_vals)), y_vals, alpha=0.15, color="#2196F3")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(y_key)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def _plot_scatter(ax, data, x_key, y_key, title):
    x_vals = [float(row.get(x_key, 0)) for row in data]
    y_vals = [float(row.get(y_key, 0)) for row in data]
    labels = [str(row.get("player", "")) for row in data]
    ax.scatter(x_vals, y_vals, alpha=0.7, s=50, c="#673AB7")
    for x, y, lbl in zip(x_vals, y_vals, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, alpha=0.8)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(linestyle="--", alpha=0.4)


def _plot_pie(ax, data, x_key, y_key, title):
    labels = [str(row.get(x_key, "?")) for row in data]
    values = [abs(float(row.get(y_key, 0))) for row in data]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        startangle=90, pctdistance=0.85
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


CHART_FUNCTIONS = {
    "bar":     _plot_bar,
    "line":    _plot_line,
    "scatter": _plot_scatter,
    "pie":     _plot_pie,
}


def _make_cache_key(data, chart_type, x_key, y_key, title) -> str:
    """
    Génère un hash MD5 court et déterministe basé sur les paramètres.
    Les mêmes paramètres produisent toujours le même nom de fichier.
    """
    payload = json.dumps(
        {"data": data, "chart_type": chart_type, "x_key": x_key, "y_key": y_key, "title": title},
        sort_keys=True
    )
    return hashlib.md5(payload.encode()).hexdigest()[:10]


def generate_chart(data, chart_type="bar", x_key="player",
                   y_key="pts", title="Statistiques NBA") -> str:
    """
    Génère un graphique et le sauvegarde sur disque.
    Si un graphique identique existe déjà (même hash), le retourne directement
    sans le régénérer.
    """
    if not data:
        raise ValueError("Aucune donnée fournie pour le graphique.")

    # Nom de fichier déterministe : titre_safe + hash des paramètres
    safe_title = "".join(c if c.isalnum() or c in "_-" else "_" for c in title)[:30]
    cache_key  = _make_cache_key(data, chart_type, x_key, y_key, title)
    filepath   = GRAPHS_DIR / f"{safe_title}_{cache_key}.png"

    # ── Cache hit : fichier déjà généré, on le retourne directement ──────────
    if filepath.exists():
        logger.info(f"Cache hit — graphique existant retourné : {filepath}")
        return str(filepath.resolve())

    # ── Cache miss : génération du graphique ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    plot_fn = CHART_FUNCTIONS.get(chart_type, _plot_bar)
    plot_fn(ax, data, x_key, y_key, title)
    plt.tight_layout()

    plt.savefig(filepath, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Graphique généré et sauvegardé : {filepath}")
    return str(filepath.resolve())


@tool
def plot_tool(plot_request: str) -> str:
    """
    Génère un graphique à partir de données NBA structurées.
    L'input doit être un JSON avec les champs :
      - data       : liste de dicts avec les données
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

    Retourne : GRAPH_FILE:/chemin/absolu/vers/image.png
    """
    logger.info("Plot Tool appelé")

    try:
        params = json.loads(plot_request)
    except json.JSONDecodeError as e:
        return f"Erreur de parsing JSON : {e}"

    for field in ["data", "x_key", "y_key", "title"]:
        if field not in params:
            return f"Champ requis manquant : {field}"

    try:
        filepath = generate_chart(
            data=params["data"],
            chart_type=params.get("chart_type", "bar"),
            x_key=params["x_key"],
            y_key=params["y_key"],
            title=params["title"],
        )
        return f"GRAPH_FILE:{filepath}"
    except Exception as e:
        logger.error(f"Erreur graphique: {e}")
        return f"Erreur lors de la génération du graphique : {e}"
