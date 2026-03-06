# evaluate_ragas.py
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

TEST_CASES_DEFINITION = [
    # ── SIMPLE ──────────────────────────────────────────────
    {
        "question": "Qui a marque le plus de points par match cette saison ?",
        "ground_truth": "Shai Gilgeous-Alexander a la meilleure moyenne de points par match.",
        "category": "simple"
    },
    {
        "question": "Quel joueur a le meilleur pourcentage de tirs a 3 points parmi les joueurs ayant tente plus de 100 tirs ?",
        "ground_truth": "Le joueur avec le meilleur 3P% parmi ceux avec fg3a >= 100.",
        "category": "simple"
    },
    {
        "question": "Combien de joueurs ont realise plus de 5 triple-doubles cette saison ?",
        "ground_truth": "Reponse basee sur le champ TD3 de la base de donnees.",
        "category": "simple"
    },
    {
        "question": "Quelle est la moyenne de points de l'equipe OKC ?",
        "ground_truth": "La moyenne des points des joueurs d'OKC.",
        "category": "simple"
    },

    # ── COMPLEX ─────────────────────────────────────────────
    {
        "question": "Compare les statistiques defensives (rebonds, contres, interceptions) de Rudy Gobert et Brook Lopez.",
        "ground_truth": "Comparaison des stats defensives entre Gobert et Lopez.",
        "category": "complex"
    },
    {
        "question": "Quelle equipe a le meilleur Net Rating moyen sur l'ensemble de ses joueurs ?",
        "ground_truth": "L'equipe avec le NETRTG moyen le plus eleve.",
        "category": "complex"
    },
    {
        "question": "Identifie les joueurs qui cumulent plus de 20 points, 8 rebonds et 5 passes decisives par match.",
        "ground_truth": "Joueurs avec PTS>20, REB>8, AST>5 simultanement.",
        "category": "complex"
    },
    {
        "question": "Quelle est la relation entre le Usage Rate et les points marques pour les joueurs de plus de 30 matchs ?",
        "ground_truth": "Analyse de la relation USG% vs PTS pour GP > 30.",
        "category": "complex"
    },

    # ── NOISY ────────────────────────────────────────────────
    {
        "question": "c ki le mieux joueur nba cette saison niveau attaque ??",
        "ground_truth": "Le meilleur attaquant de la saison malgre la formulation incorrecte.",
        "category": "noisy"
    },
    {
        "question": "Donne moi les stats du joueur numero 23",
        "ground_truth": "Question ambigue : numero 23 peut etre LeBron James.",
        "category": "noisy"
    },
    {
        "question": "Qui est le meilleur joueur de basket cette annee en termes de performance globale ?",
        "ground_truth": "Le PIE (Player Impact Estimate) est la metrique la plus adaptee.",
        "category": "noisy"
    },
    {
        "question": "Je veux savoir les stats de l'equipe de Los Angeles",
        "ground_truth": "Ambiguite : LAL (Lakers) ou LAC (Clippers) ?",
        "category": "noisy"
    },
]


def query_rag_system(question: str) -> tuple[str, list[str]]:
    try:
        from utils.vector_store import VectorStoreManager
        from utils.config import SEARCH_K
        from agent import get_agent_response

        vsm     = VectorStoreManager()
        results = vsm.search(question, k=SEARCH_K)
        contexts = [r["text"] for r in results]
        answer   = get_agent_response(question)
        return answer, contexts

    except Exception as e:
        logger.warning(f"Systeme RAG non disponible, placeholder utilise: {e}")
        return (
            f"[PLACEHOLDER] Reponse pour : {question}",
            [f"[PLACEHOLDER] Contexte 1 pour : {question}",
             f"[PLACEHOLDER] Contexte 2 pour : {question}"]
        )


def compute_ragas_scores(test_cases: list[dict]) -> list[dict]:
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from datasets import Dataset

        dataset_dict = {
            "question":     [tc["question"]              for tc in test_cases],
            "answer":       [tc["answer"]                for tc in test_cases],
            "contexts":     [tc["contexts"]              for tc in test_cases],
            "ground_truth": [tc.get("ground_truth", "") for tc in test_cases],
        }
        dataset = Dataset.from_dict(dataset_dict)

        logger.info("Lancement evaluation RAGAS...")
        result     = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                                context_precision, context_recall])
        scores_df  = result.to_pandas()

        for i, tc in enumerate(test_cases):
            row = scores_df.iloc[i]
            tc["faithfulness"]      = round(float(row.get("faithfulness",      0)), 4)
            tc["answer_relevancy"]  = round(float(row.get("answer_relevancy",  0)), 4)
            tc["context_precision"] = round(float(row.get("context_precision", 0)), 4)
            tc["context_recall"]    = round(float(row.get("context_recall",    0)), 4)
            tc["overall_score"]     = round(
                (tc["faithfulness"] + tc["answer_relevancy"] +
                 tc["context_precision"] + tc["context_recall"]) / 4, 4
            )

    except ImportError:
        logger.warning("RAGAS non installe. Scores = None.")
        for tc in test_cases:
            tc["faithfulness"] = tc["answer_relevancy"] = None
            tc["context_precision"] = tc["context_recall"] = None
            tc["overall_score"] = None
    except Exception as e:
        logger.error(f"Erreur RAGAS : {e}")
        for tc in test_cases:
            tc["error"] = str(e)

    return test_cases


def generate_report(results: list[dict], mode: str, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    report   = {
        "mode":        mode,
        "timestamp":   datetime.now().isoformat(),
        "total_cases": len(results),
        "results":     results
    }
    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Rapport JSON : {json_path}")

    categories   = ["simple", "complex", "noisy"]
    all_metrics  = ["faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "overall_score"]
    summary_lines = [
        f"{'='*60}",
        f"RAPPORT RAGAS — Mode: {mode.upper()}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'='*60}", ""
    ]

    for cat in categories:
        cat_results = [r for r in results if r.get("category") == cat]
        if not cat_results:
            continue
        summary_lines.append(f"\n── Categorie : {cat.upper()} ({len(cat_results)} cas) ──")
        for metric in all_metrics:
            scores = [r[metric] for r in cat_results if r.get(metric) is not None]
            if scores:
                avg = round(sum(scores) / len(scores), 4)
                bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
                summary_lines.append(f"  {metric:<22} {bar} {avg:.4f}")
            else:
                summary_lines.append(f"  {metric:<22} N/A")

    all_overall = [r["overall_score"] for r in results if r.get("overall_score") is not None]
    if all_overall:
        g