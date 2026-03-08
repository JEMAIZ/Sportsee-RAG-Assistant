# team_tool.py
import logging
from langchain.tools import tool
from utils.database import get_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _query(sql: str, params: dict = {}) -> list:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


@tool
def team_tool(team_name: str) -> str:
    """
    Retourne les statistiques complètes d'une équipe NBA :
    - Top 5 scoreurs, rebondeurs, passeurs
    - Roster complet avec stats clés

    L'input est le nom ou le code de l'équipe (ex: 'Lakers', 'LAL', 'Golden State', 'GSW').
    """
    logger.info(f"Team Tool appelé pour : {team_name}")

    # Recherche flexible : code exact OU full_name partiel
    team_rows = _query("""
        SELECT code, full_name
        FROM teams
        WHERE LOWER(code)      = LOWER(:code)
           OR LOWER(full_name) LIKE LOWER(:partial)
        LIMIT 1
    """, {"code": team_name, "partial": f"%{team_name}%"})

    if not team_rows:
        all_teams = _query("SELECT code, full_name FROM teams ORDER BY full_name")
        team_list = ", ".join(f"{t['full_name']} ({t['code']})" for t in all_teams)
        return f"Équipe '{team_name}' non trouvée.\nÉquipes disponibles : {team_list}"

    team      = team_rows[0]
    team_code = team["code"]
    team_full = team["full_name"]

    lines = [f"# 🏀 {team_full} ({team_code})", ""]

    # ── Top 5 scoreurs ───────────────────────────────────────────────────────
    top_scorers = _query("""
        SELECT player, pts, gp,
               ROUND(CAST(pts AS FLOAT) / NULLIF(gp, 0), 1) AS ppg
        FROM players
        WHERE team = :code AND gp > 0
        ORDER BY pts DESC LIMIT 5
    """, {"code": team_code})

    lines.append("## 🔥 Top 5 Scoreurs")
    for i, p in enumerate(top_scorers, 1):
        lines.append(f"{i}. **{p['player']}** — {p['ppg']} pts/match ({p['gp']} matchs)")

    # ── Top 5 rebondeurs ─────────────────────────────────────────────────────
    top_reb = _query("""
        SELECT player, reb, gp,
               ROUND(CAST(reb AS FLOAT) / NULLIF(gp, 0), 1) AS rpg
        FROM players
        WHERE team = :code AND gp > 0
        ORDER BY reb DESC LIMIT 5
    """, {"code": team_code})

    lines.append("\n## 💪 Top 5 Rebondeurs")
    for i, p in enumerate(top_reb, 1):
        lines.append(f"{i}. **{p['player']}** — {p['rpg']} reb/match")

    # ── Top 5 passeurs ───────────────────────────────────────────────────────
    top_ast = _query("""
        SELECT player, ast, gp,
               ROUND(CAST(ast AS FLOAT) / NULLIF(gp, 0), 1) AS apg
        FROM players
        WHERE team = :code AND gp > 0
        ORDER BY ast DESC LIMIT 5
    """, {"code": team_code})

    lines.append("\n## 🎯 Top 5 Passeurs")
    for i, p in enumerate(top_ast, 1):
        lines.append(f"{i}. **{p['player']}** — {p['apg']} ast/match")

    # ── Roster complet ───────────────────────────────────────────────────────
    roster = _query("""
        SELECT player, gp, pts, reb, ast, stl, blk, fg_pct, fg3_pct, ft_pct,
               ROUND(CAST(pts AS FLOAT) / NULLIF(gp, 0), 1) AS ppg,
               ROUND(CAST(reb AS FLOAT) / NULLIF(gp, 0), 1) AS rpg,
               ROUND(CAST(ast AS FLOAT) / NULLIF(gp, 0), 1) AS apg
        FROM players
        WHERE team = :code
        ORDER BY pts DESC
    """, {"code": team_code})

    lines.append(f"\n## 📋 Roster complet ({len(roster)} joueurs)")
    lines.append(f"{'Joueur':<25} {'GP':>4} {'PPG':>6} {'RPG':>6} {'APG':>6} {'STL':>5} {'BLK':>5} {'FG%':>6} {'3P%':>6} {'FT%':>6}")
    lines.append("-" * 82)
    for p in roster:
        fg  = f"{p['fg_pct']:.1%}"  if p['fg_pct']  else "N/A"
        fg3 = f"{p['fg3_pct']:.1%}" if p['fg3_pct'] else "N/A"
        ft  = f"{p['ft_pct']:.1%}"  if p['ft_pct']  else "N/A"
        lines.append(
            f"{str(p['player']):<25} {int(p['gp'] or 0):>4} {p['ppg']:>6} {p['rpg']:>6} "
            f"{p['apg']:>6} {int(p['stl'] or 0):>5} {int(p['blk'] or 0):>5} {fg:>6} {fg3:>6} {ft:>6}"
        )

    return "\n".join(lines)


@tool
def list_teams_tool(query: str = "") -> str:
    """
    Retourne la liste de toutes les équipes NBA disponibles dans la base.
    Utile pour connaître les noms exacts avant d'appeler team_tool.
    """
    rows = _query("SELECT code, full_name FROM teams ORDER BY full_name")
    if not rows:
        return "Aucune équipe trouvée dans la base de données."
    lines = ["# 🏀 Équipes NBA disponibles", ""]
    for t in rows:
        lines.append(f"- **{t['full_name']}** ({t['code']})")
    return "\n".join(lines)
