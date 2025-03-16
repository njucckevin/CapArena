# Calculate CapArena-Auto Scores by performing pairwise evaluation
# usage: python caparena_auto_eval.py --test_model Model-Test --result_path xxx/test_model_result.json --imgs_dir xxx/all_images
import json
import os
import random
import argparse
from tqdm import tqdm
from vlm_as_a_judge import mllm_judge_pairs


def calculate_caparena_auto_scores(test_model, caparena_eval_dir, result_path, imgs_dir, caparena_auto_600_path):

    test_model_save = os.path.join(caparena_eval_dir, f"{test_model}.json")
    # Step1: convert the caption result to caparena_auto format
    caparena_auto_600 = json.load(open(caparena_auto_600_path, 'r'))
    test_model_data = json.load(open(result_path, 'r'))
    caparena_eval = []

    for img_filename, img_data in tqdm(caparena_auto_600.items()):
        if img_data["ref_model"] == test_model:
            continue

        data_item = dict()
        data_item["img"] = img_filename

        ref_model = img_data["ref_model"]
        eval_model = test_model
        ref_caption = img_data["captions"][ref_model]
        eval_caption = test_model_data[img_filename]

        if random.randint(0, 1) == 0:
            data_item["source1"] = ref_model
            data_item["source2"] = eval_model
            data_item["caption1"] = ref_caption
            data_item["caption2"] = eval_caption
        else:
            data_item["source1"] = eval_model
            data_item["source2"] = ref_model
            data_item["caption1"] = eval_caption
            data_item["caption2"] = ref_caption

        data_item["ref"] = img_data["captions"]["human"]
        data_item["ref_model"] = ref_model

        caparena_eval.append(data_item)

    print(f"Num of eval for model {test_model}: {len(caparena_eval)}")
    json.dump(caparena_eval, open(test_model_save, "w"))

    # Step2: use GPT-4o-as-a-Judge to perform pairwise judgment for the model's generated results
    print(f"Evaluating {test_model} ...")
    mllm_judge_pairs(
        caption_eval_cand_dir=test_model_save,
        imgs_dir=imgs_dir,
        with_ref=True,
        cal_agree=False,
        eval_model_name=test_model
    )

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate CapArena-Auto Scores and perform pairwise evaluation.")
    parser.add_argument('--caparena_eval_dir', type=str, default='data/caparena_auto', help="Directory to save evaluation data.")
    parser.add_argument('--caparena_auto_600_path', type=str, default='data/caparena_auto/caparena_auto_600.json', help="Path to the CapArena Auto 600 JSON file.")
    parser.add_argument('--test_model', type=str, required=True, help="The name of the model to test.")
    parser.add_argument('--result_path', type=str, required=True, help="Path to the result JSON file.")
    parser.add_argument('--imgs_dir', type=str, required=True, help="Directory containing the images.")

    args = parser.parse_args()

    calculate_caparena_auto_scores(
        test_model=args.test_model,
        caparena_eval_dir=args.caparena_eval_dir,
        result_path=args.result_path,
        imgs_dir=args.imgs_dir,
        caparena_auto_600_path=args.caparena_auto_600_path
    )

if __name__ == "__main__":
    main()
