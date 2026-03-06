# utils/schemas.py
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class PlayerStats(BaseModel):
    player: str             = Field(..., description="Nom du joueur")
    team:   str             = Field(..., description="Code equipe 3 lettres")
    age:    Optional[float] = None
    gp:     Optional[float] = None
    w:      Optional[float] = None
    l:      Optional[float] = None
    min_pg: Optional[float] = None
    pts:    Optional[float] = None
    fgm:    Optional[float] = None
    fga:    Optional[float] = None
    fg_pct: Optional[float] = None
    fg3m:   Optional[float] = None
    fg3a:   Optional[float] = None
    fg3_pct: Optional[float] = None
    ftm:    Optional[float] = None
    fta:    Optional[float] = None
    ft_pct: Optional[float] = None
    oreb:   Optional[float] = None
    dreb:   Optional[float] = None
    reb:    Optional[float] = None
    ast:    Optional[float] = None
    tov:    Optional[float] = None
    stl:    Optional[float] = None
    blk:    Optional[float] = None
    pf:     Optional[float] = None
    fp:     Optional[float] = None
    dd2:    Optional[float] = None
    td3:    Optional[float] = None
    plus_minus: Optional[float] = None
    offrtg: Optional[float] = None
    defrtg: Optional[float] = None
    netrtg: Optional[float] = None
    ast_pct: Optional[float] = None
    ast_to:  Optional[float] = None
    ast_ratio: Optional[float] = None
    oreb_pct: Optional[float] = None
    dreb_pct: Optional[float] = None
    reb_pct:  Optional[float] = None
    to_ratio: Optional[float] = None
    efg_pct:  Optional[float] = None
    ts_pct:   Optional[float] = None
    usg_pct:  Optional[float] = None
    pace:    Optional[float] = None
    pie:     Optional[float] = None
    poss:    Optional[float] = None

    @field_validator("team")
    @classmethod
    def team_uppercase(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("player")
    @classmethod
    def player_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Nom du joueur vide.")
        return v.strip()


class TeamInfo(BaseModel):
    code:      str = Field(..., description="Code 3 lettres")
    full_name: str = Field(..., description="Nom complet")

    @field_validator("code")
    @classmethod
    def code_format(cls, v: str) -> str:
        v = v.upper().strip()
        if len(v) != 3:
            raise ValueError(f"Code equipe invalide: {v}")
        return v


class RAGChunk(BaseModel):
    chunk_id:    str            = Field(...)
    text:        str            = Field(..., min_length=10)
    source:      str            = Field(...)
    chunk_index: int            = Field(..., ge=0)
    metadata:    dict[str, Any] = Field(default_factory=dict)


class RAGEvalCase(BaseModel):
    question:     str           = Field(...)
    answer:       str           = Field(...)
    contexts:     list[str]     = Field(..., min_length=1)
    ground_truth: Optional[str] = None
    category:     str           = Field(default="simple")

    @field_validator("category")
    @classmethod
    def valid_category(cls, v: str) -> str:
        allowed = {"simple", "complex", "noisy"}
        if v not in allowed:
            raise ValueError(f"Categorie invalide: {v}. Choisir: {allowed}")
        return v


class RAGEvalResult(BaseModel):
    question:          str             = Field(...)
    category:          str             = Field(...)
    faithfulness:      Optional[float] = Field(None, ge=0, le=1)
    answer_relevancy:  Optional[float] = Field(None, ge=0, le=1)
    context_precision: Optional[float] = Field(None, ge=0, le=1)
    context_recall:    Optional[float] = Field(None, ge=0, le=1)
    error:             Optional[str]   = None

    @property
    def overall_score(self) -> Optional[float]:
        scores = [s for s in [
            self.faithfulness, self.answer_relevancy,
            self.context_precision, self.context_recall
        ] if s is not None]
        return round(sum(scores) / len(scores), 4) if scores else None


class SQLQueryResult(BaseModel):
    query:     str                  = Field(...)
    rows:      list[dict[str, Any]] = Field(default_factory=list)
    row_count: int                  = Field(0, ge=0)
    error:     Optional[str]        = None

    @model_validator(mode="after")
    def set_row_count(self) -> "SQLQueryResult":
        self.row_count = len(self.rows)
        return self