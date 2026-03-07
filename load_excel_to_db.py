# load_excel_to_db.py
import argparse
import logging
import math
from typing import Optional

import pandas as pd
from pydantic import ValidationError
from sqlalchemy.orm import Session

from utils.config import DB_PATH
from utils.database import init_db, Team, Player
from utils.schemas import PlayerStats, TeamInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def extract_teams(excel_path: str) -> list[dict]:
    logger.info("Lecture feuille Equipe...")
    df = pd.read_excel(excel_path, sheet_name="Equipe", header=0)
    df.columns = ["code", "full_name"]
    return df.dropna(subset=["code"]).to_dict("records")


def extract_players(excel_path: str) -> list[dict]:
    logger.info("Lecture feuille Donnees NBA...")
    df = pd.read_excel(excel_path, sheet_name="Données NBA", header=1)
    df = df.iloc[:, :45]
    df = df.dropna(subset=["Player"])
    return df.to_dict("records")


def transform_team(raw: dict) -> Optional[TeamInfo]:
    try:
        return TeamInfo(
            code=str(raw.get("code", "")).strip(),
            full_name=str(raw.get("full_name", "")).strip()
        )
    except ValidationError as e:
        logger.warning(f"Equipe ignoree: {raw} — {e}")
        return None


def transform_player(raw: dict) -> Optional[PlayerStats]:
    try:
        return PlayerStats(
            player=str(raw.get("Player", "")).strip(),
            team=str(raw.get("Team", "")).strip(),
            age=_safe_float(raw.get("Age")),
            gp=_safe_float(raw.get("GP")),
            w=_safe_float(raw.get("W")),
            l=_safe_float(raw.get("L")),
            min_pg=_safe_float(raw.get("Min")),
            pts=_safe_float(raw.get("PTS")),
            fgm=_safe_float(raw.get("FGM")),
            fga=_safe_float(raw.get("FGA")),
            fg_pct=_safe_float(raw.get("FG%")),
            fg3m=_safe_float(raw.get("3PM")),
            fg3a=_safe_float(raw.get("3PA")),
            fg3_pct=_safe_float(raw.get("3P%")),
            ftm=_safe_float(raw.get("FTM")),
            fta=_safe_float(raw.get("FTA")),
            ft_pct=_safe_float(raw.get("FT%")),
            oreb=_safe_float(raw.get("OREB")),
            dreb=_safe_float(raw.get("DREB")),
            reb=_safe_float(raw.get("REB")),
            ast=_safe_float(raw.get("AST")),
            tov=_safe_float(raw.get("TOV")),
            stl=_safe_float(raw.get("STL")),
            blk=_safe_float(raw.get("BLK")),
            pf=_safe_float(raw.get("PF")),
            fp=_safe_float(raw.get("FP")),
            dd2=_safe_float(raw.get("DD2")),
            td3=_safe_float(raw.get("TD3")),
            plus_minus=_safe_float(raw.get('="+ / -"') or raw.get("+/-")),
            offrtg=_safe_float(raw.get("OFFRTG")),
            defrtg=_safe_float(raw.get("DEFRTG")),
            netrtg=_safe_float(raw.get("NETRTG")),
            ast_pct=_safe_float(raw.get("AST%")),
            ast_to=_safe_float(raw.get("AST/TO")),
            ast_ratio=_safe_float(raw.get("AST RATIO")),
            oreb_pct=_safe_float(raw.get("OREB%")),
            dreb_pct=_safe_float(raw.get("DREB%")),
            reb_pct=_safe_float(raw.get("REB%")),
            to_ratio=_safe_float(raw.get("TO RATIO")),
            efg_pct=_safe_float(raw.get("EFG%")),
            ts_pct=_safe_float(raw.get("TS%")),
            usg_pct=_safe_float(raw.get("USG%")),
            pace=_safe_float(raw.get("PACE")),
            pie=_safe_float(raw.get("PIE")),
            poss=_safe_float(raw.get("POSS")),
        )
    except ValidationError as e:
        logger.warning(f"Joueur ignore: {raw.get('Player')} — {e}")
        return None


def load_teams(session: Session, teams: list[TeamInfo]) -> int:
    count = 0
    for t in teams:
        existing = session.get(Team, t.code)
        if existing is None:
            session.add(Team(code=t.code, full_name=t.full_name))
            count += 1
    session.commit()
    logger.info(f"{count} equipes inserees.")
    return count


def load_players(session: Session, players: list[PlayerStats]) -> tuple[int, int]:
    from sqlalchemy import select
    inserted = 0
    skipped  = 0
    for p in players:
        stmt     = select(Player).where(Player.player == p.player, Player.team == p.team)
        existing = session.execute(stmt).scalar_one_or_none()
        if existing is not None:
            skipped += 1
            continue
        session.add(Player(
            player=p.player, team=p.team, age=p.age,
            gp=p.gp, w=p.w, l=p.l, min_pg=p.min_pg, pts=p.pts,
            fgm=p.fgm, fga=p.fga, fg_pct=p.fg_pct,
            fg3m=p.fg3m, fg3a=p.fg3a, fg3_pct=p.fg3_pct,
            ftm=p.ftm, fta=p.fta, ft_pct=p.ft_pct,
            oreb=p.oreb, dreb=p.dreb, reb=p.reb,
            ast=p.ast, tov=p.tov, stl=p.stl, blk=p.blk, pf=p.pf,
            fp=p.fp, dd2=p.dd2, td3=p.td3, plus_minus=p.plus_minus,
            offrtg=p.offrtg, defrtg=p.defrtg, netrtg=p.netrtg,
            ast_pct=p.ast_pct, ast_to=p.ast_to, ast_ratio=p.ast_ratio,
            oreb_pct=p.oreb_pct, dreb_pct=p.dreb_pct, reb_pct=p.reb_pct,
            to_ratio=p.to_ratio, efg_pct=p.efg_pct, ts_pct=p.ts_pct,
            usg_pct=p.usg_pct, pace=p.pace, pie=p.pie, poss=p.poss,
        ))
        inserted += 1
    session.commit()
    logger.info(f"{inserted} joueurs inseres, {skipped} ignores (doublons).")
    return inserted, skipped


def run_etl(excel_path: str, db_path: str = DB_PATH):
    logger.info("Demarrage pipeline ETL SportSee")
    engine = init_db(db_path)

    raw_teams   = extract_teams(excel_path)
    raw_players = extract_players(excel_path)
    logger.info(f"Donnees brutes : {len(raw_teams)} equipes, {len(raw_players)} joueurs")

    teams_valid   = [t for raw in raw_teams   if (t := transform_team(raw))]
    players_valid = [p for raw in raw_players if (p := transform_player(raw))]
    logger.info(f"Apres validation : {len(teams_valid)} equipes, {len(players_valid)} joueurs")

    with Session(engine) as session:
        load_teams(session, teams_valid)
        load_players(session, players_valid)

    logger.info("Pipeline ETL termine !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Excel NBA -> SQLite")
    parser.add_argument("--excel-path", type=str, default="inputs/regular_NBA.xlsx")
    parser.add_argument("--db-path",    type=str, default=DB_PATH)
    args = parser.parse_args()
    run_etl(args.excel_path, args.db_path)