# Calculate CapArena-Auto Leaderboard scores
# usage: python caparena_auto_scores.py --caparena_auto_dir data/caparena_auto
# usage: python caparena_auto_scores.py --caparena_auto_dir data/caparena_auto --new_model_name Model-Test
import json
import os
import argparse

def calculate_caparena_scores(caparena_auto_dir, new_model_name=None):
    model_list_default = [
        "GPT-4o-0806", "Claude-3.5-Sonnet-0620", "Gemini-1.5-pro-002", "InternVL2-26B",
        "Gemini-2.0-flash-exp", "Qwen2-VL-72B-Instruct", "CogVLM2-llama3-chat-19B",
        "GPT-4o-mini-0718", "LLama-3.2-90B", "MiniCPM-V2.6-8B", "LLaVA-1.6-34B",
        "Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "LLaVA-1.5-7B", "cambrian-34b",
        "LLaVA-OV-72b", "Ovis-1_6-27b", "Ovis-2-34b", "Internvl2-5-8b", "Qwen2.5VL-72B",
        "Hunyuan-standard-vision", "GLM-4V-Plus"
    ]
    # model_list_default = [
    #     "GPT-4o-0806", "Claude-3.5-Sonnet-0620", "Gemini-1.5-pro-002", "InternVL2-26B",
    #     "Gemini-2.0-flash-exp", "Qwen2-VL-72B-Instruct", "CogVLM2-llama3-chat-19B",
    #     "GPT-4o-mini-0718", "LLama-3.2-90B", "MiniCPM-V2.6-8B", "LLaVA-1.6-34B",
    #     "Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "LLaVA-1.5-7B"
    # ]

    if new_model_name:
        new_model_list = [new_model_name]
    else:
        new_model_list = None
        
    model_list = model_list_default if new_model_list is None else model_list_default + new_model_list

    score_all = {}
    for model_name in model_list:
        caparena_auto_eval = json.load(open(os.path.join(caparena_auto_dir, model_name+".json"), "r"))
        score_refs = {"GPT-4o-0806": [0, 0], "CogVLM2-llama3-chat-19B": [0, 0], "MiniCPM-V2.6-8B": [0, 0]}
        caption_length = []
        for data_item in caparena_auto_eval:

            if model_name == data_item["source1"]:
                caption_length.append(len(data_item["caption1"].split(' ')))
            else:
                caption_length.append(len(data_item["caption2"].split(' ')))

            if "judge" not in data_item:
                continue

            if data_item["judge"] not in ["Caption 1 is better.", "Caption 1 is better", "Caption 2 is better.",
                                     "Caption 2 is better", "Tie", "Tie."]:
                print("GPT judgment not 1 or 2 or tie")
                continue
            score_refs[data_item["ref_model"]][0] += 1

            if data_item["judge"] in "Caption 1 is better.":
                winner = data_item["source1"]
            elif data_item["judge"] in "Caption 2 is better.":
                winner = data_item["source2"]
            else:
                winner = "Tie"

            if winner == "Tie":
                pass
            elif winner == model_name:
                score_refs[data_item["ref_model"]][1] += 1
            else:
                score_refs[data_item["ref_model"]][1] -= 1

        avg_score = 0
        for k, v in score_refs.items():
            score_refs[k][1] = score_refs[k][1]/2
            avg_score += score_refs[k][1]
        score_refs["Score_Avg"] = avg_score/3

        # Calculate average length
        score_refs["Length_Avg"] = sum(caption_length)/len(caption_length)

        score_all[model_name] = score_refs

    sorted_models = sorted(score_all.items(), key=lambda x: x[1]['Score_Avg'], reverse=True)

    print("CapArena-Auto Leaderboard:")
    print(f"{'Model':<30} | {'Score_avg':<8} | {'Score_gpt':<10} | {'Score_cog':<10} | {'Score_cpm':<10} | {'Length_Avg':<10} |")
    print("-" * 85)

    for model, data in sorted_models:
        score_gpt = data['GPT-4o-0806'][1]
        score_cog = data['CogVLM2-llama3-chat-19B'][1]
        score_cpm = data['MiniCPM-V2.6-8B'][1]

        # Format each line to align with columns
        print(f"{model:<30} | {data['Score_Avg']:<8.2f} | {score_gpt:<10.2f} | {score_cog:<10.2f} | {score_cpm:<10.2f} | {data['Length_Avg']:<10.2f} |")

    return [model for model, data in sorted_models]


def main():
    parser = argparse.ArgumentParser(description='Calculate CapArena Auto Scores')
    parser.add_argument('--caparena_auto_dir', type=str, default="data/caparena_auto", help='Directory containing CapArena auto evaluation files')
    parser.add_argument('--new_model_name', type=str, default=None, help='Name of new model to add to the leaderboard')
    args = parser.parse_args()
    
    sorted_model_names = calculate_caparena_scores(args.caparena_auto_dir, args.new_model_name)
    print("\nSorted model names:")
    print(sorted_model_names)


if __name__ == "__main__":
    main()
