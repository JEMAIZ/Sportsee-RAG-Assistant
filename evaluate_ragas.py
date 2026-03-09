# evaluate_ragas.py
"""
Évaluation RAGAS du système SportSee RAG.
En mode CI (--ci), retourne exit code 1 si le score global < RAGAS_THRESHOLD.
"""
import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Seuil CI (overridable via env var) ───────────────────────────────────────
RAGAS_THRESHOLD = float(os.getenv("RAGAS_THRESHOLD", "0.65"))

# ── Cas de test ───────────────────────────────────────────────────────────────
TEST_CASES = {
    "SIMPLE": [
        {
            "question": "Qui est le meilleur scoreur de la saison ?",
            "ground_truth": "Shai Gilgeous-Alexander est le meilleur scoreur avec environ 32 points par match."
        },
        {
            "question": "Quelle équipe a le code LAL ?",
            "ground_truth": "LAL est le code des Los Angeles Lakers."
        },
        {
            "question": "Combien d'équipes y a-t-il dans la base de données ?",
            "ground_truth": "Il y a 30 équipes NBA dans la base de données."
        },
        {
            "question": "Quel joueur a le meilleur pourcentage aux tirs à 3 points ?",
            "ground_truth": "Le joueur avec le meilleur 3P% parmi ceux ayant suffisamment de tentatives."
        },
    ],
    "COMPLEX": [
        {
            "question": "Quels joueurs combinent plus de 25 points, 7 rebonds et 6 passes par match ?",
            "ground_truth": "Les joueurs comme LeBron James, Nikola Jokic répondent à ces critères multi-statistiques."
        },
        {
            "question": "Compare l'efficacité offensive des 5 meilleures équipes selon le net rating.",
            "ground_truth": "Les équipes avec le meilleur net rating positif sont les plus efficaces offensivement."
        },
        {
            "question": "Quel est le top 5 des joueurs avec le meilleur true shooting percentage parmi ceux jouant plus de 60 matchs ?",
            "ground_truth": "Les joueurs avec ts_pct élevé parmi ceux ayant joué 60+ matchs."
        },
        {
            "question": "Quels joueurs des Lakers ont les meilleures statistiques défensives ?",
            "ground_truth": "Les joueurs des Lakers (LAL) avec le plus de blocks et steals."
        },
    ],
    "NOISY": [
        {
            "question": "c ki le mieu pour les point cette annee nba ??",
            "ground_truth": "Shai Gilgeous-Alexander est le meilleur scoreur de la saison NBA."
        },
        {
            "question": "montre moi un peux les stats des warriors stp",
            "ground_truth": "Les Golden State Warriors (GSW) ont plusieurs joueurs notables avec des statistiques variées."
        },
        {
            "question": "ya des joueur avec genre 20 point et plein de passe ?",
            "ground_truth": "Plusieurs joueurs combinent 20+ points et un nombre élevé de passes décisives."
        },
        {
            "question": "qui joue le mieu en defense dans toute la ligue",
            "ground_truth": "Les meilleurs défenseurs se distinguent par leurs blocks et steals élevés."
        },
    ],
}

SCORE_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def get_agent_answer(question: str) -> tuple[str, str]:
    """Retourne (réponse_agent, contexte_utilisé)."""
    from agent import build_agent
    from sql_tool import sql_tool

    agent = build_agent()
    result = agent.invoke({"input": question, "chat_history": []})
    answer = result.get("output", "")

    # Contexte = résultat SQL brut pour la même question
    try:
        context = sql_tool.invoke(question)
    except Exception:
        context = "Contexte non disponible"

    return answer, context


def score_with_mistral(question: str, answer: str, context: str, ground_truth: str) -> dict:
    """Évalue les 4 métriques RAGAS via Mistral."""
    from langchain_mistralai import ChatMistralAI
    from utils.config import MISTRAL_API_KEY, MODEL_NAME

    llm = ChatMistralAI(api_key=MISTRAL_API_KEY, model=MODEL_NAME, temperature=0)

    prompts = {
        "faithfulness": f"""
Évalue si la réponse est fidèle au contexte fourni (pas d'hallucination).
Contexte: {context[:500]}
Réponse: {answer[:300]}
Donne UNIQUEMENT un score entre 0.0 et 1.0 (ex: 0.85). Rien d'autre.""",

        "answer_relevancy": f"""
Évalue si la réponse est pertinente par rapport à la question.
Question: {question}
Réponse: {answer[:300]}
Donne UNIQUEMENT un score entre 0.0 et 1.0 (ex: 0.85). Rien d'autre.""",

        "context_precision": f"""
Évalue si le contexte récupéré est précis et utile pour répondre.
Question: {question}
Contexte: {context[:500]}
Donne UNIQUEMENT un score entre 0.0 et 1.0 (ex: 0.85). Rien d'autre.""",

        "context_recall": f"""
Évalue si le contexte contient les informations nécessaires pour répondre correctement.
Question: {question}
Vérité terrain: {ground_truth}
Contexte: {context[:500]}
Donne UNIQUEMENT un score entre 0.0 et 1.0 (ex: 0.85). Rien d'autre.""",
    }

    scores = {}
    for metric, prompt in prompts.items():
        try:
            response = llm.invoke(prompt)
            raw = response.content.strip().replace(",", ".")
            # Extrait le premier nombre flottant trouvé
            import re
            match = re.search(r"\d+\.\d+|\d+", raw)
            score = float(match.group()) if match else 0.5
            scores[metric] = min(1.0, max(0.0, score))
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Erreur scoring {metric}: {e}")
            scores[metric] = 0.5

    scores["overall_score"] = sum(scores[m] for m in SCORE_METRICS) / len(SCORE_METRICS)
    return scores


def run_evaluation(mode: str = "baseline", ci_mode: bool = False) -> float:
    """Lance l'évaluation complète. Retourne le score global."""
    logger.info(f"Démarrage évaluation RAGAS — mode={mode}, ci={ci_mode}")

    all_results = {}
    category_scores = {}

    for category, cases in TEST_CASES.items():
        logger.info(f"\n── Catégorie : {category} ({len(cases)} cas) ──")
        cat_results = []

        for i, case in enumerate(cases):
            logger.info(f"  [{i+1}/{len(cases)}] {case['question'][:60]}...")
            answer, context = get_agent_answer(case["question"])
            scores = score_with_mistral(
                case["question"], answer, context, case["ground_truth"]
            )
            cat_results.append({
                "question":    case["question"],
                "answer":      answer,
                "ground_truth": case["ground_truth"],
                "scores":      scores,
            })
            logger.info(f"  → overall={scores['overall_score']:.4f}")
            time.sleep(2)

        # Moyennes par catégorie
        avg = {}
        for m in SCORE_METRICS + ["overall_score"]:
            avg[m] = sum(r["scores"][m] for r in cat_results) / len(cat_results)

        category_scores[category] = avg
        all_results[category]     = cat_results

    # ── Score global ──────────────────────────────────────────────────────────
    global_score = sum(s["overall_score"] for s in category_scores.values()) / len(category_scores)

    # ── Rapport texte ─────────────────────────────────────────────────────────
    bar_width = 20
    report_lines = ["=" * 60, f"  RAPPORT RAGAS — mode: {mode}", "=" * 60, ""]

    for cat, scores in category_scores.items():
        n = len(TEST_CASES[cat])
        report_lines.append(f"── Catégorie : {cat} ({n} cas) ──")
        for m in SCORE_METRICS + ["overall_score"]:
            v     = scores[m]
            filled = int(v * bar_width)
            bar   = "█" * filled + "░" * (bar_width - filled)
            report_lines.append(f"  {m:<25} {bar} {v:.4f}")
        report_lines.append("")

    # Indicateur CI
    status = "✅ PASS" if global_score >= RAGAS_THRESHOLD else "❌ FAIL"
    report_lines += [
        "=" * 60,
        f"  SCORE GLOBAL : {global_score:.4f}",
        f"  SEUIL CI     : {RAGAS_THRESHOLD}",
        f"  STATUT       : {status}",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    Path("outputs").mkdir(exist_ok=True)
    suffix = f"_{mode}"
    with open(f"outputs/ragas{suffix}.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(f"outputs/ragas{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode":            mode,
            "global_score":    global_score,
            "threshold":       RAGAS_THRESHOLD,
            "passed":          global_score >= RAGAS_THRESHOLD,
            "category_scores": category_scores,
            "details":         all_results,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Rapports sauvegardés dans outputs/ragas{suffix}.*")

    # ── Exit code CI ──────────────────────────────────────────────────────────
    if ci_mode and global_score < RAGAS_THRESHOLD:
        logger.error(
            f"RAGAS score {global_score:.4f} < seuil {RAGAS_THRESHOLD} — pipeline bloqué."
        )
        sys.exit(1)

    return global_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation RAGAS SportSee")
    parser.add_argument("--mode",      default="baseline",
                        choices=["baseline", "post-sql"],
                        help="Mode d'évaluation")
    parser.add_argument("--ci",        action="store_true",
                        help="Mode CI : exit 1 si score < seuil")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Seuil override (défaut: {RAGAS_THRESHOLD})")
    args = parser.parse_args()

    if args.threshold is not None:
        RAGAS_THRESHOLD = args.threshold

    run_evaluation(mode=args.mode, ci_mode=args.ci)
