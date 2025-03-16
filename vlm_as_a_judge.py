# usage: python vlm_as_a_judge.py --caption_eval_cand_dir data/eval/caparena_annots_eval.json --eval_save_path data/eval/caparena_annots_eval_gpt_ref.json --imgs_dir data/eval/images
import os
import time
import json
import requests
import base64
from tqdm import tqdm
import re
import argparse
import shutil


# Helper functions for image encoding
def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')


def convert_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        return encode_image(image_bytes)


# Function to call OpenAI API
def call_llm(model_name, payload):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }
    print("Generating content with GPT model: {}".format(model_name))
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json={**payload, "temperature": 0.1}
    )
    if response.status_code != 200:
        if response.json()['error']['code'] == "context_length_exceeded":
            print("Context length exceeded. Retrying with a smaller context.")
            payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
            retry_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={**payload, "temperature": 0.1}
            )
            if retry_response.status_code != 200:
                print(
                    "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                return ""

        print("Failed to call LLM: " + response.text)
        time.sleep(2)
        return ""
    else:
        return response.json()['choices'][0]['message']['content']


# Calculate agreement between model predictions and human judgments
def cal_agreement(caption_eval_pair_list, include_tie=False, in_400=False):
    agreement_level = {"overall": [], "level 1": [], "level 2": [], "level 3": [], "level 4": []}
    tie_num = {"level 1": 0, "level 2": 0, "level 3": 0, "level 4": 0}
    for item in caption_eval_pair_list:
        if "judge" not in item:
            continue

        if item["source1"] == "human" or item["source2"] == "human":
           continue

        if in_400 == True:
            if not item["in-400"]:
                continue

        if item["judge"] not in ["Caption 1 is better.", "Caption 1 is better", "Caption 2 is better.", "Caption 2 is better", "Tie", "Tie."]:
            print("GPT judgment not 1 or 2")
            continue

        if item["winner"] == item["source1"]:
            judge_human = "Caption 1 is better."
        elif item["winner"] == item["source2"]:
            judge_human = "Caption 2 is better."
        else:
            judge_human = "Tie."

        if not include_tie:
            if judge_human != "Tie." and item["judge"] != "Tie." and item["judge"] != "Tie":
                agree = 1 if item["judge"] in judge_human else 0
                agreement_level["overall"].append(agree)
                agreement_level[item["cluster"]].append(agree)
        else:
            agree = 1 if item["judge"] in judge_human else 0
            agreement_level["overall"].append(agree)
            agreement_level[item["cluster"]].append(agree)

            if item["judge"] == "Tie" or item["judge"] == "Tie.":
                tie_num[item["cluster"]] += 1

    overall = sum(agreement_level["overall"]) / len(agreement_level["overall"]) if len(
        agreement_level["overall"]) > 0 else None
    level1 = sum(agreement_level["level 1"]) / len(agreement_level["level 1"]) if len(
        agreement_level["level 1"]) > 0 else None
    level2 = sum(agreement_level["level 2"]) / len(agreement_level["level 2"]) if len(
        agreement_level["level 2"]) > 0 else None
    level3 = sum(agreement_level["level 3"]) / len(agreement_level["level 3"]) if len(
        agreement_level["level 3"]) > 0 else None
    level4 = sum(agreement_level["level 4"]) / len(agreement_level["level 4"]) if len(
        agreement_level["level 4"]) > 0 else None
    overall_num = len(agreement_level["overall"])
    level1_num = len(agreement_level["level 1"])
    level2_num = len(agreement_level["level 2"])
    level3_num = len(agreement_level["level 3"])
    level4_num = len(agreement_level["level 4"])

    result = (
        f"Overall: {overall if overall is None else f'{overall:.3f}'} ({overall_num}), "
        f"Level 1: {level1 if level1 is None else f'{level1:.3f}'} ({level1_num}), "
        f"Level 2: {level2 if level2 is None else f'{level2:.3f}'} ({level2_num}), "
        f"Level 3: {level3 if level3 is None else f'{level3:.3f}'} ({level3_num}), "
        f"Level 4: {level4 if level4 is None else f'{level4:.3f}'} ({level4_num})"
    )
    print(result)

    if include_tie:
        level1_num = tie_num["level 1"]
        level2_num = tie_num["level 2"]
        level3_num = tie_num["level 3"]
        level4_num = tie_num["level 4"]
        result_tie_num = f"Level 1: {level1_num}, Level 2: {level2_num}, Level 3: {level3_num}, Level 4: {level4_num}"
        print(result_tie_num)


system_prompt_without_ref = """
You are a highly capable multimodal AI assistant tasked with evaluating image captions.

Given an image and two candidate captions, you are require to determine which of the two captions is better.

Below are some guidelines for your reference:

1. **Precision**: The caption should accurately correspond to the content of the image, providing precise information about it. Common examples of imprecision include errors in color, quantity, spatial relationships, or the posture of people.

2. **Informativeness**: Salient information in the image should be reflected in the caption. Since it is impossible to include every detail, you will need to subjectively judge which aspects of the image are important. For instance, describing an otter as "a small animal" is precise, but it is less informative than specifying "an otter".

3. **Hallucination**: Captions that include descriptions of objects or elements that are clearly absent from the image should be significantly penalized.

4. **Attention to detail**: Annotators should pay close attention to the details in the image to distinguish the quality of the descriptions.

5. **Assistive description**: Imagine a visually impaired person asking you to describe the image for them. How would you convey the image to them?

6. **Reverse thinking**: What image does the caption lead us to imagine? Does the caption effectively lead you to imagine the intended image?

7. **Ties are acceptable**: If you find it genuinely difficult to determine which caption is better (e.g., both captions are excellent), marking a tie is acceptable.

While the above guidelines provide a framework, they cannot cover all possible cases. Therefore, we encourage you to make **subjective judgments** based on the specific circumstances and your own reasoning about which caption is better.

### Response Format:
Format your response into two lines as shown below:
Reason: <your thoughts and reasoning process for the judgment>
Judgment: <Caption 1 is better>/<Caption 2 is better>/<Tie>
"""

system_prompt_with_ref = """
You are a highly capable multimodal AI assistant tasked with evaluating image captions.

Given an image, two candidate captions and one reference caption annotated by human expert, you are require to determine which of the two captions is better.

Below are some guidelines for your reference:

1. **Precision**: The caption should accurately correspond to the content of the image, providing precise information about it. Common examples of imprecision include errors in color, quantity, spatial relationships, or the posture of people.

2. **Informativeness**: Salient information in the image should be reflected in the caption. Since it is impossible to include every detail, you will need to subjectively judge which aspects of the image are important. For instance, describing an otter as "a small animal" is precise, but it is less informative than specifying "an otter".

3. **Hallucination**: Captions that include descriptions of objects or elements that are clearly absent from the image should be significantly penalized.

4. **Attention to detail**: Annotators should pay close attention to the details in the image to distinguish the quality of the descriptions.

5. **Assistive description**: Imagine a visually impaired person asking you to describe the image for them. How would you convey the image to them?

6. **Reverse thinking**: What image does the caption lead us to imagine? Does the caption effectively lead you to imagine the intended image?

7. **Ties are acceptable**: If you find it genuinely difficult to determine which caption is better (e.g., both captions are excellent), marking a tie is acceptable.

While the above guidelines provide a framework, they cannot cover all possible cases. Therefore, we encourage you to make **subjective judgments** based on the specific circumstances and your own reasoning about which caption is better.

**Reference caption**: The reference caption is annotated by a human expert. When you're uncertain about which description is better (e.g., when unsure about specific details in the image), you can use the reference caption to assist your judgment. The content in the reference caption can be considered correct; however, it is not perfect, and descriptions not included in the reference caption can still be reasonable.

### Response Format:
Format your response into two lines as shown below:
Reason: <your thoughts and reasoning process for the judgment>
Judgment: <Caption 1 is better>/<Caption 2 is better>/<Tie>
"""

def mllm_judge_pairs(caption_eval_cand_dir, imgs_dir, with_ref=True, cal_agree=True, eval_model_name=None):

    caption_eval_cand = json.load(open(caption_eval_cand_dir, 'r'))
    print(f"Num of All Caption Pair: {len(caption_eval_cand)}")

    for i, item in tqdm(enumerate(caption_eval_cand)):

        if "judge" in item:
            print("processed")
            continue

        if i % 20 == 0:
            json.dump(caption_eval_cand, open(caption_eval_cand_dir, 'w'))

        if item["source1"] == "human" or item["source2"] == "human":
            continue

        # if not item["in-400"]:
        #     continue

        img_filename = item["img"]
        img_path = os.path.join(imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print("img not exist")
        image = convert_image_to_base64(img_path)

        caption_1 = item["caption1"]
        caption_2 = item["caption2"]
        caption_ref = item["ref"]

        if with_ref:
            compare_prompt = f"Caption 1:\n{caption_1}\nCaption 2:\n{caption_2}\nCaption Reference:\n{caption_ref}\nDetermine which is better and answer with the given format. Only mark a tie if it is truly difficult to decide which caption is better based on their quality, informativeness, and precision."
        else:
            compare_prompt = f"Caption 1:\n{caption_1}\nCaption 2:\n{caption_2}\nDetermine which is better and answer with the given format. Only mark a tie if there is no discernible difference in quality, informativeness, and precision after careful evaluation."

        messages = []

        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt_with_ref if with_ref else system_prompt_without_ref
                },
            ]
        })

        action_text_image = []
        action_text_image.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}",
                    "detail": "high"
                }
            }
        )
        action_text_image.append(
            {
                "type": "text",
                "text": compare_prompt
            }
        )

        messages.append({
            "role": "user",
            "content": action_text_image
        })

        print(compare_prompt)

        model_name = "gpt-4o-2024-08-06"
        try_num = 0
        while try_num < 5:
            try_num += 1
            try:
                response = call_llm(model_name, {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 1500,
                    "top_p": 0.9,
                    "temperature": 0.5
                })
            except:
                print("error call")
                time.sleep(1.0)
                continue
            try:
                print(response)
                reason_match = re.search(r"Reason:\s*(.+?)\s*Judgment:", response, re.DOTALL)
                judge_match = re.search(r"Judgment:\s*(.+)", response)
                reason = reason_match.group(1).strip() if reason_match else None
                judgment = judge_match.group(1).strip() if judge_match else None

                if reason and judgment:
                    item["judge_reason"] = response
                    item["judge"] = judgment
                    break
                else:
                    print("Invalid response format, retrying...")
                    time.sleep(1.0)

            except json.JSONDecodeError:
                # If response is not valid JSON, continue generating
                print("Invalid response received, retrying...")
                time.sleep(1.0)

        num_processed = len([item for item in caption_eval_cand if ("judge" in item)])
        if eval_model_name != None:
            print("Eval Model: {} Num of total: {} Num of success: {}".format(eval_model_name, len(caption_eval_cand), num_processed))
        else:
            print("Num of total: {} Num of success: {}".format(len(caption_eval_cand), num_processed))

        if cal_agree:
            cal_agreement(caption_eval_cand, include_tie=True)

    json.dump(caption_eval_cand, open(caption_eval_cand_dir, 'w'))
    print("Done")

def main():
    parser = argparse.ArgumentParser(description='Evaluate the quality of image captions')
    parser.add_argument('--caption_eval_cand_dir', type=str, required=True, help='Path to JSON file containing caption evaluation candidates')
    parser.add_argument('--eval_save_path', type=str, required=True, help='Path to save evaluation results')
    parser.add_argument('--imgs_dir', type=str, required=True, help='Path to directory containing images')
    parser.add_argument('--with_ref', type=bool, default=True, help='Whether to use reference captions for evaluation')
    parser.add_argument('--cal_agree', type=bool, default=True, help='Whether to calculate agreement')
    parser.add_argument('--eval_model_name', type=str, default=None, help='Name of evaluation model')

    args = parser.parse_args()
    # Copy original evaluation file to new save path
    shutil.copy(args.caption_eval_cand_dir, args.eval_save_path)
    print(f"Evaluation file copied to: {args.eval_save_path}")

    mllm_judge_pairs(
        caption_eval_cand_dir=args.eval_save_path,
        imgs_dir=args.imgs_dir,
        with_ref=args.with_ref,
        cal_agree=args.cal_agree,
        eval_model_name=args.eval_model_name
    )

if __name__ == "__main__":
    main()