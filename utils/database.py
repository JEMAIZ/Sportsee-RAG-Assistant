# utils/database.py
import os
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, Session

from utils.config import DB_PATH

Base = declarative_base()


class Team(Base):
    __tablename__ = "teams"

    code      = Column(String(3), primary_key=True)
    full_name = Column(String(100), nullable=False)
    players   = relationship("Player", back_populates="team_ref")

    def __repr__(self):
        return f"<Team {self.code} — {self.full_name}>"


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (
        UniqueConstraint("player", "team", name="uq_player_team"),
    )

    id       = Column(Integer, primary_key=True, autoincrement=True)
    player   = Column(String(100), nullable=False, index=True)
    team     = Column(String(3), ForeignKey("teams.code"), nullable=False, index=True)
    age      = Column(Float)
    gp       = Column(Float)
    w        = Column(Float)
    l        = Column(Float)
    min_pg   = Column(Float)
    pts      = Column(Float)
    fgm      = Column(Float)
    fga      = Column(Float)
    fg_pct   = Column(Float)
    fg3m     = Column(Float)
    fg3a     = Column(Float)
    fg3_pct  = Column(Float)
    ftm      = Column(Float)
    fta      = Column(Float)
    ft_pct   = Column(Float)
    oreb     = Column(Float)
    dreb     = Column(Float)
    reb      = Column(Float)
    ast      = Column(Float)
    tov      = Column(Float)
    stl      = Column(Float)
    blk      = Column(Float)
    pf       = Column(Float)
    fp       = Column(Float)
    dd2      = Column(Float)
    td3      = Column(Float)
    plus_minus = Column(Float)
    offrtg   = Column(Float)
    defrtg   = Column(Float)
    netrtg   = Column(Float)
    ast_pct  = Column(Float)
    ast_to   = Column(Float)
    ast_ratio = Column(Float)
    oreb_pct = Column(Float)
    dreb_pct = Column(Float)
    reb_pct  = Column(Float)
    to_ratio = Column(Float)
    efg_pct  = Column(Float)
    ts_pct   = Column(Float)
    usg_pct  = Column(Float)
    pace     = Column(Float)
    pie      = Column(Float)
    poss     = Column(Float)
    team_ref = relationship("Team", back_populates="players")

    def __repr__(self):
        return f"<Player {self.player} ({self.team}) — {self.pts} pts>"


def get_engine(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: str = DB_PATH):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine