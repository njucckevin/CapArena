import json
import csv
from collections import defaultdict
import json, math, gdown
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

pd.options.display.float_format = '{:.2f}'.format


def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, source1, source2, winner in battles[['source1', 'source2', "winner_gpt"]].itertuples():
        ra = rating[source1]
        rb = rating[source2]
        # if "human" in [source1, source2]:
        #     continue
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == source1:
            sa = 1
        elif winner == source2:
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)" or winner == "equal":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[source1] += K * (sa - ea)
        rating[source2] += K * (1 - sa - eb)

    return rating


def preety_print_elo_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def compute_elo_mle(df, SCALE=400, BASE=10, INIT_RATING=1000):
    from sklearn.linear_model import LogisticRegression
    models = pd.concat([df["source1"], df["source2"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["source1"]]] = +math.log(BASE)
    X[np.arange(n), models[df["source2"]]] = -math.log(BASE)

    Y = np.zeros(n)
    Y[df["winner_gpt"] == df["source1"]] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower=df.quantile(.025),
        rating=df.quantile(.5),
        upper=df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus", text="rating_rounded",
                     title=title)
    print(list(bars["model"]), bars)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating")
    return fig


def calculate_elo_rankings(json_path):
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)  # data is a list of dictionaries

    result = []

    for item in data:
        if "human" in [item["source1"], item["source2"]]:
            continue
        if "judge" not in item:
            continue
        if "1" in item["judge"]:
            item["winner_gpt"] = item["source1"]
        elif "2" in item["judge"]:
            item["winner_gpt"] = item["source2"]
        else:
            item["winner_gpt"] = "equal"
        result.append(item)

    # Convert JSON data to DataFrame
    df = pd.DataFrame(result)

    # Extract required columns
    battles = df[['img', 'source1', 'source2', 'winner_gpt']]

    # # Save as CSV file
    # battles.to_csv('check.csv', index=False, encoding='utf-8')

    # Calculate ELO scores
    elo_ratings = compute_elo(battles)
    preety_print_elo_ratings(elo_ratings)

    # Calculate bootstrap results
    BOOTSTRAP_ROUNDS = 1000
    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(battles, compute_elo, BOOTSTRAP_ROUNDS)
    bootstrap_lu_median = bootstrap_elo_lu.median().reset_index().set_axis(["model", "Elo rating"], axis=1)
    bootstrap_lu_median["Elo rating"] = (bootstrap_lu_median["Elo rating"] + 0.5).astype(int)
    
    print("Elo ranking by metrics:")
    print(bootstrap_lu_median)
    
    return list(bootstrap_lu_median["model"])


if __name__ == "__main__":
    sorted_model_names = calculate_elo_rankings('data/eval/caparena_annots_eval_qwen25vl72b.json')
    print(sorted_model_names)
