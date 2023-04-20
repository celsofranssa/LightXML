import pickle

import numpy as np
import pandas as pd
import scipy
import torch
from ranx import evaluate, Run, Qrels
from torch.utils.data import DataLoader
import tqdm

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


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return f"{round(100 * m, 1)}({round(100 * h, 1)})"


def get_ic(model, dataset):
    print(f"IC for {model} - {dataset}")
    result_df = pd.read_csv(f"resource/result/{model}_{dataset}.rts", header=0, sep="\t")
    for cls in ["tail", "head"]:
        print(f"Results for {cls}")
        cls_df = result_df[result_df["cls"] == cls]
        for metric in ['mrr@1', 'mrr@5', 'mrr@10', 'ndcg@1', 'ndcg@5', 'ndcg@10']:
            print(f"{metric}: {mean_confidence_interval(cls_df[metric])}")


def eval(dataset, folds):
    results = []
    text_cls = load_text_cls(dataset)
    label_cls = load_label_cls(dataset)
    metrics = get_metrics(["mrr", "recall", "ndcg", "precision", "hit_rate"], [1, 5, 10])

    for fold_idx in folds:
        print(f"Evaluating fold {fold_idx}")
        df, label_map = createDataCSV(dataset, fold=fold_idx)
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
        for cls in ["all", "head", "tail"]:
            relevance_map = {}
            ranking = {}
            for index, labels in enumerate(df.label.values):
                labels = filter_labels(labels, cls, label_cls)
                if cls in text_cls[texts_map[index]]:
                    true_labels = set([label_map[i] for i in labels.split()])
                    t = {}
                    for true_label in true_labels:
                        t[f"label_{true_label}"] = 1.0
                    relevance_map[f"text_{index}"] = t

                    logits = [torch.sigmoid(predicts[i][index]) for i in range(len(berts))]
                    logits.append(sum(logits))
                    logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

                    p = {}
                    for pst, pred_label in enumerate(logits[-1]):
                        p[f"label_{pred_label}"] = 1.0 / (pst + 1)
                    ranking[f"text_{index}"] = p if len(p) > 0 else {"label_-1": 0.0}

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
            results.append(result)

    checkpoint_results(results, dataset, model="LightXML")
    print(f"Results\n{pd.DataFrame(results)}\n")


def direct_ensemble(dataset, folds):
    results = []
    text_cls = load_text_cls(dataset)
    label_cls = load_label_cls(dataset)
    metrics = get_metrics(["mrr", "recall", "ndcg", "precision", "hit_rate"], [1, 5, 10])

    for fold_idx in folds:
        print(f"Evaluating fold {fold_idx}")

        model_labels, model_scores = [], []
        models = [f"{dataset}_t0", f"{dataset}_t1", f"{dataset}_t2"]
        for model in models:
            print(f'loading {model}')
            model_scores.append(np.load(f'./resource/prediction/fold_{fold_idx}/{model}-scores.npy', allow_pickle=True))
            model_labels.append(np.load(f'./resource/prediction/fold_{fold_idx}/{model}-labels.npy', allow_pickle=True))

        df, label_map = createDataCSV(dataset, fold=fold_idx)
        print(f'load {dataset} dataset with '
              f'{len(df[df.dataType == "train"])} train {len(df[df.dataType == "test"])} test with {len(label_map)} labels done')

        df = df[df.dataType == 'test']
        texts_map = get_texts_map(dataset, fold_idx=fold_idx, split="test")
        print(f"Evaluating")
        for cls in ["all", "head", "tail"]:
            relevance_map = {}
            ranking = {}
            for index, labels in enumerate(df.label.values):
                labels = filter_labels(labels, cls, label_cls)
                if cls in text_cls[texts_map[index]]:
                    true_labels = set([label_map[i] for i in labels.split()])
                    t = {}
                    for true_label in true_labels:
                        t[f"label_{true_label}"] = 1.0
                    relevance_map[f"text_{index}"] = t

                    pred_labels = {}
                    for j in range(len(models)):
                        for l, s in sorted(list(zip(model_labels[j][index], model_scores[j][index])), key=lambda x: x[1],
                                           reverse=True):
                            if l in pred_labels:
                                pred_labels[l] += s
                            else:
                                pred_labels[l] = s

                    pred_labels = [k for k, v in sorted(pred_labels.items(), key=lambda item: item[1], reverse=True)]

                    p = {}
                    for pst, pred_label in enumerate(pred_labels):
                        p[f"label_{pred_label}"] = 1.0 / (pst + 1)
                    ranking[f"text_{index}"] = p if len(p) > 0 else {"label_-1": 0.0}

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
            results.append(result)


    checkpoint_results(results, dataset, model="LightXML")
    print(f"Results\n{pd.DataFrame(results)}\n")




if __name__ == '__main__':
    dataset = "Amazon-670k"
    # direct_ensemble(dataset, folds=[0,1])
    # eval(dataset, folds=[0, 1, 2, 3, 4])

    get_ic("LightXML", dataset)
