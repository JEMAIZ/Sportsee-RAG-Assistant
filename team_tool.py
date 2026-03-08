# team_tool.py
import logging
from langchain.tools import tool
from utils.database import get_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _query(sql: str, params: dict = {}) -> list[dict]:
    """Exécute une requête SQL et retourne une liste de dicts."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


@tool
def team_tool(team_name: str) -> str:
    """
    Retourne les statistiques complètes d'une équipe NBA :
    - Informations générales de l'équipe
    - Top 5 scoreurs de l'équipe
    - Top 5 rebondeurs de l'équipe
    - Top 5 passeurs de l'équipe
    - Listing complet des joueurs avec leurs stats clés

    L'input est le nom ou l'abréviation de l'équipe (ex: 'Lakers', 'LAL', 'Golden State', 'GSW').
    """
    logger.info(f"Team Tool appelé pour : {team_name}")

    # ── Recherche flexible de l'équipe ───────────────────────────────────────
    team_rows = _query("""
        SELECT team_id, team_name, abbreviation
        FROM teams
        WHERE LOWER(team_name)     LIKE LOWER(:q)
           OR LOWER(abbreviation)  LIKE LOWER(:q)
           OR LOWER(team_name)     LIKE LOWER(:partial)
        LIMIT 1
    """, {"q": team_name, "partial": f"%{team_name}%"})

    if not team_rows:
        # Liste toutes les équipes disponibles
        all_teams = _query("SELECT team_name, abbreviation FROM teams ORDER BY team_name")
        team_list = ", ".join(f"{t['team_name']} ({t['abbreviation']})" for t in all_teams)
        return f"Équipe '{team_name}' non trouvée. Équipes disponibles : {team_list}"

    team      = team_rows[0]
    team_id   = team["team_id"]
    team_full = team["team_name"]
    team_abbr = team["abbreviation"]

    result_lines = [
        f"# 🏀 {team_full} ({team_abbr})",
        ""
    ]

    # ── Top 5 scoreurs ───────────────────────────────────────────────────────
    top_scorers = _query("""
        SELECT player, pts, gp,
               ROUND(CAST(pts AS FLOAT) / NULLIF(gp, 0), 1) AS pts_per_game
        FROM players
        WHERE team_id = :tid AND gp > 0
        ORDER BY pts DESC
        LIMIT 5
    """, {"tid": team_id})

    result_lines.append("## 🔥 Top 5 Scoreurs")
    for i, p in enumerate(top_scorers, 1):
        result_lines.append(
            f"{i}. **{p['player']}** — {p['pts']} pts totaux "
            f"({p['pts_per_game']} pts/match, {p['gp']} matchs)"
        )

    # ── Top 5 rebondeurs ─────────────────────────────────────────────────────
    top_reb = _query("""
        SELECT player, reb, gp,
               ROUND(CAST(reb AS FLOAT) / NULLIF(gp, 0), 1) AS reb_per_game
        FROM players
        WHERE team_id = :tid AND gp > 0
        ORDER BY reb DESC
        LIMIT 5
    """, {"tid": team_id})

    result_lines.append("\n## 💪 Top 5 Rebondeurs")
    for i, p in enumerate(top_reb, 1):
        result_lines.append(
            f"{i}. **{p['player']}** — {p['reb']} reb totaux "
            f"({p['reb_per_game']} reb/match)"
        )

    # ── Top 5 passeurs ───────────────────────────────────────────────────────
    top_ast = _query("""
        SELECT player, ast, gp,
               ROUND(CAST(ast AS FLOAT) / NULLIF(gp, 0), 1) AS ast_per_game
        FROM players
        WHERE team_id = :tid AND gp > 0
        ORDER BY ast DESC
        LIMIT 5
    """, {"tid": team_id})

    result_lines.append("\n## 🎯 Top 5 Passeurs")
    for i, p in enumerate(top_ast, 1):
        result_lines.append(
            f"{i}. **{p['player']}** — {p['ast']} ast totales "
            f"({p['ast_per_game']} ast/match)"
        )

    # ── Roster complet ───────────────────────────────────────────────────────
    roster = _query("""
        SELECT player, gp, pts, reb, ast, stl, blk,
               fg_pct, fg3_pct, ft_pct,
               ROUND(CAST(pts AS FLOAT) / NULLIF(gp, 0), 1) AS ppg,
               ROUND(CAST(reb AS FLOAT) / NULLIF(gp, 0), 1) AS rpg,
               ROUND(CAST(ast AS FLOAT) / NULLIF(gp, 0), 1) AS apg
        FROM players
        WHERE team_id = :tid
        ORDER BY pts DESC
    """, {"tid": team_id})

    result_lines.append(f"\n## 📋 Roster complet ({len(roster)} joueurs)")
    result_lines.append(
        f"{'Joueur':<25} {'GP':>4} {'PPG':>6} {'RPG':>6} {'APG':>6} "
        f"{'STL':>5} {'BLK':>5} {'FG%':>6} {'3P%':>6} {'FT%':>6}"
    )
    result_lines.append("-" * 80)

    for p in roster:
        fg  = f"{p['fg_pct']:.1%}"  if p['fg_pct']  else "N/A"
        fg3 = f"{p['fg3_pct']:.1%}" if p['fg3_pct'] else "N/A"
        ft  = f"{p['ft_pct']:.1%}"  if p['ft_pct']  else "N/A"
        result_lines.append(
            f"{str(p['player']):<25} {p['gp']:>4} {p['ppg']:>6} {p['rpg']:>6} "
            f"{p['apg']:>6} {p['stl']:>5} {p['blk']:>5} {fg:>6} {fg3:>6} {ft:>6}"
        )

    return "\n".join(result_lines)


@tool
def list_teams_tool(query: str = "") -> str:
    """
    Retourne la liste de toutes les équipes NBA disponibles dans la base.
    Utile pour connaître les noms exacts avant d'appeler team_tool.
    L'input peut être vide ou un filtre partiel (ex: 'east', 'west', 'LA').
    """
    rows = _query("""
        SELECT team_name, abbreviation
        FROM teams
        ORDER BY team_name
    """)
    if not rows:
        return "Aucune équipe trouvée dans la base de données."

    lines = ["# 🏀 Équipes NBA disponibles", ""]
    for t in rows:
        lines.append(f"- **{t['team_name']}** ({t['abbreviation']})")
    return "\n".join(lines)
