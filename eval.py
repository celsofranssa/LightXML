import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from ranx import evaluate, Run, Qrels
from torch.utils.data import DataLoader
from src.dataset import MDataset, createDataCSV

from src.model import LightXML


def load_label_cls(dataset):
    with open(f"./resource/dataset/{dataset}/label_cls.pkl", "rb") as label_cls_file:
        return pickle.load(label_cls_file)


def load_text_cls(dataset):
    with open(f"./resource/dataset/{dataset}/text_cls.pkl", "rb") as text_cls_file:
        return pickle.load(text_cls_file)


def get_metrics(metrics, thresholds):
    threshold_metrics = []
    for metric in metrics:
        for threshold in thresholds:
            threshold_metrics.append(f"{metric}@{threshold}")
    return threshold_metrics


def load_ids(dataset, fold_idx, split):
    with open(f"./resource/dataset/{dataset}/fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
        return pickle.load(ids_file)


def load_samples(dataset, fold_idx, split):
    split_ids = load_ids(dataset, fold_idx, split)
    with open(f"./resource/dataset/{dataset}/samples.pkl", "rb") as samples_file:
        samples = pickle.load(samples_file)
    samples_df = pd.DataFrame(samples)
    return samples_df[samples_df["idx"].isin(split_ids)]


def get_texts_map(dataset, fold_idx, split):
    samples_df = load_samples(dataset, fold_idx, split).reset_index(drop=True)
    return pd.Series(
        samples_df["text_idx"].values,
        index=samples_df.index
    ).to_dict()


def filter_labels(labels, cls, label_cls):
    filtered_labels = ""
    for label in labels.split():
        if cls in label_cls[int(label)]:
            filtered_labels = filtered_labels + " " + label
    return filtered_labels


def checkpoint_results(results, dataset, model):
    pd.DataFrame(results).to_csv(
        f"./resource/result/{model}_{dataset}.rts",
        sep='\t', index=False, header=True)


def checkpoint_time(times, dataset, model="LightXML"):
    with open(f"resource/time/{dataset}_prediction_time.txt", "w") as f:
        for time in times:
            f.write(f"{time}\n")


def checkpoint_ranking(ranking, model, dataset, fold_idx):
    ranking_dir = f"resource/ranking/{model}_{dataset}/"
    Path(ranking_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving ranking {fold_idx} on {ranking_dir}")
    with open(f"{ranking_dir}{model}_{dataset}_{fold_idx}.rnk", "wb") as ranking_file:
        pickle.dump(ranking, ranking_file)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return f"{round(100 * m, 1)}({round(100 * h, 1)})"


def get_ic(model, dataset, metrics, thresholds):
    print(f"Eval results for {model} in {dataset}:")
    result_df = pd.read_csv(f"resource/result/{model}_{dataset}.rts", header=0, sep="\t")
    s = ""
    for metric in metrics:
        for cls in ["tail", "head"]:
            cls_df = result_df[result_df["cls"] == cls]
            print([f"{metric}@{t}" for t in thresholds])
            for m in [f"{metric}@{t}" for t in thresholds]:  # ['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@19']:
                v = mean_confidence_interval(cls_df[m])
                s = s + f"{v}\t"
    print(s)


def load_labels_map(dataset):
    labels_map = {}
    with open(f"./resource/dataset/{dataset}/samples.pkl", "rb") as samples_file:
        samples = pickle.load(samples_file)
        for sample in samples:
            for labels_idx, label in zip(sample["labels_ids"], sample["labels"]):
                label = label.replace("\n", "")
                labels_map[label] = labels_idx

    return labels_map


# def eval(dataset, folds):
def get_result(dataset, folds, metrics, thresholds):
    results = []
    rankings = {}
    text_cls = load_text_cls(dataset)
    label_cls = load_label_cls(dataset)
    metrics = get_metrics(metrics, thresholds)
    relevance_map = load_relevance_map(dataset)

    for fold_idx in folds:
        rankings[fold_idx] = {}
        print(f"Evaluating fold {fold_idx}")
        df, label_map = createDataCSV(dataset, fold=fold_idx)
        t_label_map = load_labels_map(dataset)
        predicts = []
        berts = ['bert-base', 'roberta', 'xlnet']

        print("Predicting")
        for index in range(len(berts)):
            model_name = [dataset, '' if berts[index] == 'bert-base' else berts[index]]
            model_name = '_'.join([i for i in model_name if i != ''])

            model = LightXML(n_labels=len(label_map), bert=berts[index])

            print(f'Loading ./resource/model_checkpoint/fold_{fold_idx}/model-{model_name}.bin')
            model.load_state_dict(torch.load(f'./resource/model_checkpoint/fold_{fold_idx}/model-{model_name}.bin'))

            tokenizer = model.get_tokenizer()
            test_d = MDataset(df, 'test', tokenizer, label_map,
                              128 if dataset == 'Amazoncat-13k' and berts[index] == 'xlnent' else 512)
            testloader = DataLoader(test_d, batch_size=16, num_workers=0,
                                    shuffle=False)

            model.cuda()
            predicts.append(
                torch.Tensor(model.one_epoch(0, testloader, None, mode='test')[0])
            )

        df = df[df.dataType == 'test']
        texts_map = get_texts_map(dataset, fold_idx=fold_idx, split="test")

        print(f"Evaluating")
        for cls in ["tail", "head"]:
            # relevance_map = {}
            rankings[fold_idx][cls] = {}
            ranking = {}
            for index, labels in enumerate(df.label.values):
                labels = filter_labels(labels, cls, label_cls)
                text_idx = texts_map[index]
                if cls in text_cls[text_idx]:
                    true_labels = set([label_map[i] for i in labels.split()])
                    # t = {}
                    # for true_label in true_labels:
                    #     t[f"label_{true_label}"] = 1.0
                    # relevance_map[f"text_{index}"] = t

                    logits = [torch.sigmoid(predicts[i][index]) for i in range(len(berts))]
                    logits.append(sum(logits))
                    logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

                    p = {}
                    for pst, pred_label in enumerate(logits[-1]):
                        p[f"label_{pred_label}"] = 1.0 / (pst + 1)
                    ranking[f"text_{text_idx}"] = p if len(p) > 0 else {"label_-1": 0.0}

            result = evaluate(
                Qrels(
                    {key: value for key, value in relevance_map.items() if key in ranking.keys()}
                ),
                Run(ranking),
                metrics
            )
            result = {k: round(v, 3) for k, v in result.items()}
            result["fold"] = fold_idx
            result["cls"] = cls
            rankings[fold_idx][cls] = ranking
            results.append(result)
        checkpoint_ranking(rankings[fold_idx], model="LightXML", dataset=dataset, fold_idx=fold_idx)


    checkpoint_results(results, dataset, model="LightXML")
    print(f"Results\n{pd.DataFrame(results)}\n")


def load_relevance_map(dataset):
    with open(f"resource/dataset/{dataset}/relevance_map.pkl", "rb") as relevances_file:
        data = pickle.load(relevances_file)
    relevance_map = {}
    for text_idx, labels_ids in data.items():
        d = {}
        for label_idx in labels_ids:
            d[f"label_{label_idx}"] = 1.0
        relevance_map[f"text_{text_idx}"] = d
    return relevance_map


def load_ranking(model, dataset, fold_idx):
    ranking_dir = "resource/ranking/"
    with open(f"{ranking_dir}"
              f"{model}_{dataset}/"
              f"{model}_{dataset}"
              f"_{fold_idx}.rnk", "rb") as ranking_file:
        return pickle.load(ranking_file)


def map_ranking(ranking, texts_map):
    mapped_ranking = {}
    for text_idx, labels_scores in ranking.items():
        text_idx = texts_map[int(text_idx.split("_")[-1])]
        mapped_ranking[f"text_{text_idx}"] = labels_scores

    return mapped_ranking


def eval_ranking(model, dataset, folds, eval_metrics, thresholds):
    results = []
    relevance_map = load_relevance_map(dataset)

    metrics = get_metrics(eval_metrics, thresholds)
    for fold_idx in folds:
        texts_map = get_texts_map(dataset, fold_idx=fold_idx, split="test")
        for cls in ["head", "tail"]:
            logging.info(f"Evaluating {cls} ranking")
            ranking = load_ranking(model, dataset, fold_idx)
            ranking = ranking[cls]
            ranking = map_ranking(ranking, texts_map)
            result = evaluate(
                Qrels(
                    {key: value for key, value in relevance_map.items() if key in ranking.keys()}
                ),
                Run(ranking),
                metrics
            )
            result = {k: round(v, 3) for k, v in result.items()}
            result["fold_idx"] = fold_idx
            result["cls"] = cls
            results.append(result)

    checkpoint_results(results, dataset, model)


def re_ranker(dataset, model, folds, eval_metrics, thresholds):
    results = []
    for fold_idx in folds:
        rankings = {}
        texts_map = get_texts_map(dataset, fold_idx, split="test")
        relevance_map = load_relevance_map(dataset)
        metrics = get_metrics(eval_metrics, thresholds)

        #
        labels_cls = load_label_cls(dataset)
        texts_cls = load_text_cls(dataset)

        _, label_map = createDataCSV(dataset, fold=fold_idx)

        inv_label_map = {}
        for label, label_idx in label_map.items():
            inv_label_map[label_idx] = int(label)

        tail_head_ranking = load_ranking(model, dataset, fold_idx)

        for cls in ["tail", "head"]:
            new_ranking = {}
            for tex_idx, labels_scores in tail_head_ranking["tail"].items():
                new_tex_idx = texts_map[int(tex_idx.split("_")[-1])]
                if cls in texts_cls[new_tex_idx]:
                    if f"text_{new_tex_idx}" not in new_ranking:
                        new_ranking[f"text_{new_tex_idx}"] = {}
                    for label_idx, score in labels_scores.items():
                        label_idx = int(label_idx.split("_")[-1])
                        if label_idx in inv_label_map:
                            new_label_idx = inv_label_map[label_idx]
                            if cls in labels_cls[new_label_idx]:
                                new_ranking[f"text_{new_tex_idx}"][f"label_{new_label_idx}"] = score
                    if len(new_ranking[f"text_{new_tex_idx}"]) == 0:
                        new_ranking[f"text_{new_tex_idx}"]["label_-1"] = 0.0

            for tex_idx, labels_scores in tail_head_ranking["head"].items():
                new_tex_idx = texts_map[int(tex_idx.split("_")[-1])]
                if cls in texts_cls[new_tex_idx]:
                    if f"text_{new_tex_idx}" not in new_ranking:
                        new_ranking[f"text_{new_tex_idx}"] = {}
                    for label_idx, score in labels_scores.items():
                        label_idx = int(label_idx.split("_")[-1])
                        if label_idx in inv_label_map:
                            new_label_idx = inv_label_map[label_idx]
                            if cls in labels_cls[new_label_idx]:
                                new_ranking[f"text_{new_tex_idx}"][f"label_{new_label_idx}"] = score
                    if len(new_ranking[f"text_{new_tex_idx}"]) == 0:
                        new_ranking[f"text_{new_tex_idx}"]["label_-1"] = 0.0

            result = evaluate(
                Qrels(
                    {key: value for key, value in relevance_map.items() if key in new_ranking.keys()}
                ),
                Run(new_ranking),
                metrics
            )
            result = {k: round(v, 3) for k, v in result.items()}
            result["fold_idx"] = fold_idx
            result["cls"] = cls
            results.append(result)
            rankings[cls] = new_ranking

        checkpoint_ranking(rankings, model="LightXML", dataset=dataset, fold_idx=fold_idx)

    print(pd.DataFrame(results))
    checkpoint_results(results, dataset, model="LightXML")
    print(f"Results\n{pd.DataFrame(results)}\n")

def get_time_stats(dataset):
    print(dataset)
    times = []

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return f"{round(m, 5)}({round(h, 5)})"

    predict_time = 0
    with open(f"resource/time/{dataset}_prediction_time.txt") as time_file:
        for time in time_file:
            time = time.strip()

            s, e = time.split(",")
            s = datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
            e = datetime.strptime(e.strip(), "%Y-%m-%d %H:%M:%S")

            times.append((e - s).total_seconds() / 3600)

    print(mean_confidence_interval(times))



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluation script for a model on a dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., AmazonCat-13k)')
    parser.add_argument('--folds', nargs='+', type=int, default=[5], help='List of folds')
    parser.add_argument('--metrics', nargs='+', default=["ndcg", "precision"], help='List of metrics to evaluate (default: ["ndcg", "precision"])')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[1, 3, 5, 10], help='List of thresholds for metrics (default: [1, 3, 5, 10])')



    args = parser.parse_args()

    # Use the arguments
    dataset = args.dataset
    folds = args.folds
    metrics = args.metrics
    thresholds = args.thresholds

    get_result(dataset, folds=folds, metrics=metrics, thresholds=thresholds)



if __name__ == '__main__':
    main()

