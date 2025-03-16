# CapArena: Benchmarking and Analyzing Detailed Image Captioning in the LLM Era

[![arXiv](https://img.shields.io/badge/arXiv-2503.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2503.xxxxx) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

The data, code, and resource for the paper: [CapArena: Benchmarking and Analyzing Detailed Image Captioning in the LLM Era](https://arxiv.org/abs/2503.xxxxx)

Release Plans:

- [x] Usage of *CapArena-Auto*
- [x] Code and data to reproduce the results in the paper
- [ ] Other resources

***
###  Automated Detailed Captioning Benchmark *CapArena-Auto*

*CapArena-Auto* is an arena-style automated evaluation benchmark for detailed captioning. It includes 600 evaluation images and assesses model performance through pairwise battles with three baseline models. The final score is calculated by GPT4o-as-a-Judge.

The current leaderboard contains 22 models evaluated by authors: [🤗 CapArena-Auto Leaderboard](https://huggingface.co/spaces/yan111222/CapArena_Auto)

### 📝 Evaluate your own model on *CapArena-Auto*

#### Step1：View the current leaderboard
Download the [results](https://box.nju.edu.cn/f/707c01ccdb724d2f925f/) of the models we have evaluated and put them under `data/caparena_auto`. It should looks like:

```
data/
└── caparena_auto/
    ├── GPT-4o-0806.json
    ├── Claude-3.5-Sonnet-0620.json
    ├── CogVLM2-llama3-chat-19B.json
    └── ...
```

Then, you can use `python caparena_auto_scores.py` to view the current leaderboard.

```
Model                          | Score_avg | Score_gpt  | Score_cog  | Score_cpm  | Length_Avg |
-------------------------------------------------------------------------------------
Gemini-1.5-pro-002             | 56.17    | 29.00      | 61.00      | 78.50      | 168.56     |
GPT-4o-0806                    | 44.00    | 0.00       | 55.50      | 76.50      | 115.80     |
Qwen2.5VL-72B                  | 35.33    | -1.00      | 49.00      | 58.00      | 163.67     |
Gemini-2.0-flash-exp           | 30.83    | -2.00      | 39.50      | 55.00      | 416.99     |
Ovis-2-34b                     | 27.00    | -15.00     | 33.50      | 62.50      | 120.20     |
Claude-3.5-Sonnet-0620         | 21.50    | -14.00     | 30.00      | 48.50      | 147.93     |
InternVL2-26B                  | 13.00    | -38.50     | 20.00      | 57.50      | 236.32     |
GPT-4o-mini-0718               | 9.33     | -36.00     | 17.00      | 47.00      | 139.83     |
Ovis-1_6-27b                   | 3.00     | -49.50     | 14.50      | 44.00      | 94.16      |
GLM-4V-Plus                    | -0.17    | -51.50     | 13.00      | 38.00      | 109.27     |
CogVLM2-llama3-chat-19B        | -8.50    | -56.50     | 0.00       | 31.00      | 115.87     |
Qwen2-VL-72B-Instruct          | -9.00    | -50.50     | -4.50      | 28.00      | 114.45     |
LLaVA-OV-72b                   | -12.33   | -57.50     | -6.00      | 26.50      | 200.88     |
LLama-3.2-90B                  | -25.67   | -72.00     | -13.00     | 8.00       | 160.25     |
Hunyuan-standard-vision        | -26.00   | -63.00     | -19.00     | 4.00       | 354.10     |
Internvl2-5-8b                 | -29.83   | -71.00     | -29.00     | 10.50      | 117.77     |
MiniCPM-V2.6-8B                | -38.00   | -80.00     | -34.00     | 0.00       | 106.74     |
Qwen2-VL-2B-Instruct           | -48.67   | -86.00     | -49.50     | -10.50     | 116.84     |
Qwen2-VL-7B-Instruct           | -49.00   | -78.00     | -59.00     | -10.00     | 97.81      |
LLaVA-1.6-34B                  | -67.50   | -92.00     | -53.50     | -57.00     | 124.81     |
cambrian-34b                   | -75.00   | -93.00     | -76.00     | -56.00     | 120.23     |
LLaVA-1.5-7B                   | -94.00   | -99.50     | -92.00     | -90.50     | 74.38      |
```

#### Step2: Generating captions for your own model
Download the [600 images](https://box.nju.edu.cn/f/a79c42c9c10e4acb83e7/) for *CapArena-Auto* evaluation and a [result templete](https://box.nju.edu.cn/f/43eb761488734c638824/).

Then, you should generate detailed captions for these images use your own model. The final result file is a `dict()` looks like:

```
{"test_01258.jpg": "The image features a cle… day with good weather.",
"test_04765.jpg": "The image shows a small,…on the brick structure.",
"test_02788.jpg": "The scene depicts a pair… and recycling efforts.",
"test_02765.jpg": "The photo captures a str…al beauty to the scene.",
...
```

#### Step3: Evaluating with GPT4o-as-a-Judge

First config your openai api key: 

```
export OPENAI_API_KEY="sk-xxxx"
```

Then:
```
python caparena_auto_eval.py --test_model Model-Test --result_path xxx/test_model_result.json --imgs_dir xxx/all_images
```

- test_model: a model name assigned by yourself.
- result_path: the formatted result file generated in step 2.
- imgs_dir: the 600 images directory.

This script generate the pair-wise judgment file `data/caparean_auto/Model-Test.json`.

#### Step4: View your model's score in the *CapArena-Auto* leaderboard

```
python caparena_auto_scores.py --caparena_auto_dir data/caparena_auto --new_model_name Model-Test
```

> Note: If you would like to submit your results to the [online leaderboard](https://huggingface.co/spaces/yan111222/CapArena_Auto), please raise an issue or contact us!

***
### Reproduce the results from the paper

### 🛠️ Prepare the human annotation results of *CapArena*

Download the [annotation result](https://box.nju.edu.cn/f/0fd0a0d3dce243ab8c12/) of *CapArena* and put them under `data/eval`. It should looks like:

```
data/
└── eval/
    ├── caparena_annots_eval.json
    ├── caparena_annots_eval_gpt_ref.json
    ├── caparena_annots_eval_gpt.json
    └── ...
```

`caparena_annots_eval.json` is the human annotation results of *CapArena*, which contains 6523 pair-wise battle/judgment given by our human annotators.

Other files are the results of the annotation of these 6523 pairs by captioning metrics (e.g., GPT-4o, GPT-4o with ref, LLaVA-OneVision). Each item in these files include a `judge` key to represent the judgment given by the metric.

### 🎯 Calculate the caption-level agreement and model-level agreement

Calculate caption-level agreement and model-level agreement based on metrics annotation results:

```
python caparena_metrics.py --eval_dir data/eval/caparena_annots_eval_gpt_ref.json
```

### ⚖️ VLM-as-a-Judge
The above provides the VLM-as-a-Judge results that we have generated.
If you want to reproduce our VLM-as-a-Judge process, first download the total [5100 images](https://box.nju.edu.cn/f/9d2b9ded47d54999926c/) from DOCCI.
Then you can conduct GPT4o-as-a-Judge by:
```
python vlm_as_a_judge.py --caption_eval_cand_dir data/eval/caparena_annots_eval.json --eval_save_path data/eval/caparena_annots_eval_gpt_ref.json --imgs_dir xxx/images
```

*** 
### Acknowledge

Thanks to [DOCCI](https://google.github.io/docci/) for their high-quality human annotation and wonderful open-sourced work.

Thanks to all the annotators who participated in compiling our CapArena dataset.

***
### Citation
If you find this work helpful, please consider to star 🌟 this repo and cite our paper.
```
TBD
```