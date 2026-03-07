# evaluate_ragas.py
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

TEST_CASES_DEFINITION = [
    {
        "question": "Qui a marque le plus de points cette saison ?",
        "ground_truth": "Shai Gilgeous-Alexander a la meilleure moyenne de points.",
        "category": "simple"
    },
    {
        "question": "Quel joueur a le meilleur pourcentage a 3 points avec plus de 100 tentatives ?",
        "ground_truth": "Le joueur avec le meilleur 3P% parmi ceux avec fg3a >= 100.",
        "category": "simple"
    },
    {
        "question": "Combien de joueurs ont realise plus de 5 triple-doubles cette saison ?",
        "ground_truth": "Reponse basee sur le champ TD3.",
        "category": "simple"
    },
    {
        "question": "Quelle est la moyenne de points de l'equipe OKC ?",
        "ground_truth": "La moyenne des points des joueurs d'OKC.",
        "category": "simple"
    },
    {
        "question": "Compare les stats defensives de Rudy Gobert et Brook Lopez.",
        "ground_truth": "Comparaison des stats defensives entre Gobert et Lopez.",
        "category": "complex"
    },
    {
        "question": "Quelle equipe a le meilleur Net Rating moyen ?",
        "ground_truth": "L'equipe avec le NETRTG moyen le plus eleve.",
        "category": "complex"
    },
    {
        "question": "Identifie les joueurs avec plus de 20 pts, 8 reb et 5 ast par match.",
        "ground_truth": "Joueurs avec PTS>20, REB>8, AST>5.",
        "category": "complex"
    },
    {
        "question": "Relation entre Usage Rate et points pour les joueurs de plus de 30 matchs.",
        "ground_truth": "Analyse USG% vs PTS pour GP > 30.",
        "category": "complex"
    },
    {
        "question": "c ki le mieux joueur nba cette saison niveau attaque ??",
        "ground_truth": "Le meilleur attaquant malgre la formulation incorrecte.",
        "category": "noisy"
    },
    {
        "question": "Donne moi les stats du joueur numero 23",
        "ground_truth": "Numero 23 peut etre LeBron James.",
        "category": "noisy"
    },
    {
        "question": "Qui est le meilleur joueur cette annee en performance globale ?",
        "ground_truth": "Le PIE est la metrique la plus adaptee.",
        "category": "noisy"
    },
    {
        "question": "Je veux les stats de l'equipe de Los Angeles",
        "ground_truth": "LAL (Lakers) ou LAC (Clippers) ?",
        "category": "noisy"
    },
]


def query_rag_system(question: str) -> tuple:
    try:
        from agent import get_agent_response
        answer   = get_agent_response(question)
        contexts = [f"Reponse generee par l'agent pour : {question}"]
        return answer, contexts
    except Exception as e:
        logger.warning(f"Agent non disponible: {e}")
        return (
            f"[PLACEHOLDER] Reponse pour : {question}",
            [f"[PLACEHOLDER] Contexte pour : {question}"]
        )


def eval_score(llm, prompt: str) -> float:
    """Appelle Mistral et extrait un score entre 0 et 1."""
    try:
        from langchain_core.messages import HumanMessage
        time.sleep(2)  # Evite le rate limit
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip().split()[0].replace(",", ".")
        return round(min(max(float(text), 0), 1), 4)
    except Exception as e:
        logger.warning(f"Score non calcule: {e}")
        return 0.0


def compute_ragas_scores(test_cases: list) -> list:
    try:
        from langchain_mistralai import ChatMistralAI
        from utils.config import MISTRAL_API_KEY, MODEL_NAME

        llm = ChatMistralAI(
            api_key=MISTRAL_API_KEY,
            model=MODEL_NAME,
            temperature=0
        )

        for i, tc in enumerate(test_cases):
            logger.info(f"[{i+1}/{len(test_cases)}] Evaluation: {tc['question'][:50]}...")

            question     = tc["question"]
            answer       = tc["answer"]
            contexts     = "\n".join(tc["contexts"])
            ground_truth = tc.get("ground_truth", "")

            tc["faithfulness"] = eval_score(llm, f"""Note de 0 a 1 : la reponse est-elle fidele aux contextes ?
Contextes : {contexts}
Reponse : {answer}
Reponds UNIQUEMENT avec un nombre (ex: 0.85)""")

            tc["answer_relevancy"] = eval_score(llm, f"""Note de 0 a 1 : la reponse repond-elle bien a la question ?
Question : {question}
Reponse : {answer}
Reponds UNIQUEMENT avec un nombre (ex: 0.85)""")

            tc["context_precision"] = eval_score(llm, f"""Note de 0 a 1 : les contextes sont-ils pertinents pour la question ?
Question : {question}
Contextes : {contexts}
Reponds UNIQUEMENT avec un nombre (ex: 0.85)""")

            tc["context_recall"] = eval_score(llm, f"""Note de 0 a 1 : les contextes contiennent-ils les infos pour repondre ?
Question : {question}
Reponse attendue : {ground_truth}
Contextes : {contexts}
Reponds UNIQUEMENT avec un nombre (ex: 0.85)""")

            tc["overall_score"] = round(
                (tc["faithfulness"] + tc["answer_relevancy"] +
                 tc["context_precision"] + tc["context_recall"]) / 4, 4
            )
            logger.info(f"  → F:{tc['faithfulness']} AR:{tc['answer_relevancy']} CP:{tc['context_precision']} CR:{tc['context_recall']} | Overall:{tc['overall_score']}")

    except Exception as e:
        logger.error(f"Erreur evaluation : {e}")
        for tc in test_cases:
            tc["error"] = str(e)
            tc["faithfulness"] = tc["answer_relevancy"] = None
            tc["context_precision"] = tc["context_recall"] = None
            tc["overall_score"] = None

    return test_cases


def generate_report(results: list, mode: str, output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    report = {
        "mode":        mode,
        "timestamp":   datetime.now().isoformat(),
        "total_cases": len(results),
        "results":     results
    }
    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Rapport JSON : {json_path}")

    categories  = ["simple", "complex", "noisy"]
    all_metrics = ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall", "overall_score"]

    summary_lines = [
        f"{'='*60}",
        f"RAPPORT EVALUATION — Mode: {mode.upper()}",
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
        global_avg = round(sum(all_overall) / len(all_overall), 4)
        summary_lines += [
            f"\n{'='*60}",
            f"  SCORE GLOBAL : {global_avg:.4f}",
            f"{'='*60}"
        ]

    summary = "\n".join(summary_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Resume : {output_path}")
    print(summary)
    return summary


def run_evaluation(mode: str = "baseline", report_path: Optional[str] = None):
    logger.info(f"Demarrage evaluation — mode: {mode}")
    output_path = report_path or f"outputs/ragas_{mode}.txt"
    Path("outputs").mkdir(exist_ok=True)

    test_cases = []
    for tc_def in TEST_CASES_DEFINITION:
        logger.info(f"Question [{tc_def['category']}]: {tc_def['question'][:60]}...")
        answer, contexts = query_rag_system(tc_def["question"])
        test_cases.append({**tc_def, "answer": answer, "contexts": contexts})

    test_cases = compute_ragas_scores(test_cases)
    generate_report(test_cases, mode, output_path)
    return test_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation — SportSee")
    parser.add_argument("--mode",        choices=["baseline", "post-sql"], default="baseline")
    parser.add_argument("--report-path", type=str, default=None)
    args = parser.parse_args()
    run_evaluation(mode=args.mode, report_path=args.report_path)