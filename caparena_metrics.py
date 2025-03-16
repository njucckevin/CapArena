# Calculate caption-level agreement and model-level agreement based on metrics annotation results
# Usage: python caparena_metrics.py --eval_dir data/eval/caparena_annots_eval_gpt_ref.json
import json
from cal_ranking import calculate_elo_rankings
from vlm_as_a_judge import cal_agreement
from scipy.stats import spearmanr, kendalltau
import argparse


def cal_model_level_agreement(sorted_model_names, ranking_human=["GPT-4o-0806", "human", "Gemini-2.0-flash-exp", "InternVL2-26B", "Gemini-1.5-pro-002",
                                                                "Claude-3.5-Sonnet-0620", "GPT-4o-mini-0718", "LLama-3.2-90B", "Qwen2-VL-72B-Instruct", 
                                                                "CogVLM2-llama3-chat-19B", "MiniCPM-V2.6-8B", "Qwen2-VL-7B-Instruct", "Qwen2-VL-2B-Instruct",
                                                                "LLaVA-1.6-34B", "LLaVA-1.5-7B"]):
    print(f"Num models: {len(ranking_human)}")
    print("Human ranking:")
    print(ranking_human)

    if "human" in sorted_model_names:
        sorted_model_names.remove("human")
    print("Metrics ranking:")
    print(sorted_model_names)
    sorted_ranking = [i+1 for i in range(len(sorted_model_names))]  # Model ranking positions

    # Convert ranking_human to rankings
    human_ranking = [ranking_human.index(model) + 1 for model in sorted_model_names]

    # Calculate Spearman correlation coefficient
    rho, p_value = spearmanr(human_ranking, sorted_ranking)
    print(f"Spearman ρ: {rho}")

    # Calculate Kendall Tau correlation coefficient
    tau, kendall_p_value = kendalltau(human_ranking, sorted_ranking)
    print(f"Kendall Tau: {tau}")


def cal_metrics_agreement(eval_dir):
    metrics_annot = json.load(open(eval_dir, 'r'))

    # Calculate caption-level agreement
    print("Caption-level agreement:")
    cal_agreement(metrics_annot, include_tie=True, in_400=False)

    # Calculate Elo ranking
    print("Model-level agreement:")
    sorted_model_names = calculate_elo_rankings(eval_dir)
    print(sorted_model_names)

    # Calculate model-level agreement
    cal_model_level_agreement(sorted_model_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate metrics agreement')
    parser.add_argument('--eval_dir', type=str, required=True, help='Path to JSON file containing caption evaluation candidates')
    args = parser.parse_args()
    eval_dir = args.eval_dir
    cal_metrics_agreement(eval_dir)

