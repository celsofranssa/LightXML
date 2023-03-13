import os
import torch
import pickle
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

import tqdm

def load_ids(dataset, fold_idx, split):
    with open(f"./resource/dataset/{dataset}/fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
        return pickle.load(ids_file)


def prepare_data(dataset, fold_idx, split):
    ids = load_ids(dataset, fold_idx, split)
    with open(f"./resource/dataset/{dataset}/samples.pkl", "rb") as samples_file:
        samples_df = pd.DataFrame(pickle.load(samples_file))
    samples_df = samples_df[samples_df["idx"].isin(ids)]
    # if split == "train":
    #     checkpoint_sparse_samples(samples_df, dataset)
    checkpoint_samples(samples_df, dataset, split)

# def checkpoint_sparse_samples(samples_df, dataset):
#     logger.info('Preprocessing sparse features')
#     X = TfidfVectorizer().fit_transform(samples_df["text"])
#     y = MultiLabelBinarizer(sparse_output=True).fit_transform(samples_df["labels_ids"])
#     dump_svmlight_file(X, y=y, f=f"resource/dataset/{dataset}/train_v1.txt", multilabel=True)


def checkpoint_samples(samples_df, dataset, split):
    print('Preprocessing raw texts.')
    dataset_dir = f"./resource/dataset/{dataset}/"
    samples_df["text"] = samples_df["text"].apply(lambda text: text.replace("\n", " "))
    samples_df["text"].to_csv(f"{dataset_dir}{split}_raw_texts.txt", header=False, index=False)

    samples_df["labels_ids"] = samples_df["labels_ids"].apply(lambda labels_ids: " ".join([str(idx) for idx in labels_ids]))
    samples_df["labels_ids"].to_csv(f"{dataset_dir}{split}_labels.txt", header=False, index=False)

def createDataCSV(dataset, fold, eval=False):
    labels = []
    texts = []
    dataType = []
    label_map = {}

    # name_map = {'Wiki-31k': 'Wiki10-31k',
    #             'Amazon-670k': 'Amazon-670K',
    #             'eurlex4k': 'Eurlex-4K'}
    #
    # assert dataset in name_map
    # dataset = name_map[dataset]

    if not eval:
        print('Preparing splits')
        prepare_data(dataset, fold, "train")
        prepare_data(dataset, fold, "test")

    #fext = '_texts.txt' if dataset == 'Eurlex-4K' else '_raw_texts.txt'
    data_dir = f"./resource/dataset/{dataset}/"
    with open(f"{data_dir}train_raw_texts.txt") as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('train')

    with open(f'{data_dir}test_raw_texts.txt') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('test')

    with open(f'{data_dir}train_labels.txt') as f:
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))


    with open(f'{data_dir}test_labels.txt') as f:
        print(len(label_map))
        for i in tqdm.tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))
        print(len(label_map))

    assert len(texts) == len(labels) == len(dataType)

    df_row = {'text': texts, 'label': labels, 'dataType': dataType}

    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k] = i
    df = pd.DataFrame(df_row)

    print('label map', len(label_map))

    return df, label_map


class MDataset(Dataset):
    def __init__(self, df, mode, tokenizer, label_map, max_length,
                 token_type_ids=None, group_y=None, candidates_num=None):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.df, self.n_labels, self.label_map = df[df.dataType == self.mode], len(label_map), label_map
        self.len = len(self.df)
        self.tokenizer, self.max_length, self.group_y = tokenizer, max_length, group_y
        self.multi_group = False
        self.token_type_ids = token_type_ids
        self.candidates_num = candidates_num

        if group_y is not None:
            # group y mode
            self.candidates_num, self.group_y, self.n_group_y_labels = candidates_num, [], group_y.shape[0]
            self.map_group_y = np.empty(len(label_map), dtype=np.long)
            for idx, labels in enumerate(group_y):
                self.group_y.append([])
                for label in labels:
                    self.group_y[-1].append(label_map[label])
                self.map_group_y[self.group_y[-1]] = idx
                self.group_y[-1]  = np.array(self.group_y[-1])
            self.group_y = np.array(self.group_y)

    def __getitem__(self, idx):
        max_len = self.max_length
        review = self.df.text.values[idx].lower()
        labels = [self.label_map[i] for i in self.df.label.values[idx].split() if i in self.label_map]

        review = ' '.join(review.split()[:max_len])

        text = review
        if self.token_type_ids is not None:
            input_ids = self.token_type_ids[idx]
            if input_ids[-1] == 0:
                input_ids = input_ids[input_ids != 0]
            input_ids = input_ids.tolist()
        elif hasattr(self.tokenizer, 'encode_plus'):
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True,
                max_length=max_len
            )
        else:
            # fast 
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True
            ).ids

        if len(input_ids) == 0:
            print('zero string')
            assert 0
        if len(input_ids) > self.max_length:
            input_ids[self.max_length-1] = input_ids[-1]
            input_ids = input_ids[:self.max_length]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)


        if self.group_y is not None:
            label_ids = torch.zeros(self.n_labels)
            label_ids = label_ids.scatter(0, torch.tensor(labels),
                                          torch.tensor([1.0 for i in labels]))
            group_labels = self.map_group_y[labels]
            if self.multi_group:
                group_labels = np.concatenate(group_labels)
            group_label_ids = torch.zeros(self.n_group_y_labels)
            group_label_ids = group_label_ids.scatter(0, torch.tensor(group_labels),
                                      torch.tensor([1.0 for i in group_labels]))
            candidates = np.concatenate(self.group_y[group_labels], axis=0)

            if len(candidates) < self.candidates_num:
                sample = np.random.randint(self.n_group_y_labels, size=self.candidates_num - len(candidates))
                candidates = np.concatenate([candidates, sample])
            elif len(candidates) > self.candidates_num:
                candidates = np.random.choice(candidates, self.candidates_num, replace=False)

            if self.mode == 'train':
                return input_ids, attention_mask, token_type_ids,\
                    label_ids[candidates], group_label_ids, candidates
            else:
                return input_ids, attention_mask, token_type_ids,\
                    label_ids, group_label_ids, candidates
            # root fc layer
            group_labels = layers_group_labels[0]
            group_label_ids = torch.zeros(len(self.map_children[0]))
            group_label_ids = group_label_ids.scatter(0, torch.tensor(group_labels),
                                                      torch.tensor([1.0 for i in group_labels]))
            layers_group_labels_ids.append(group_label_ids)

            if self.mode == 'train':
                return input_ids, attention_mask, token_type_ids, \
                    layers_group_labels_ids[::-1], layers_candidates[::-1]
            else:
                return input_ids, attention_mask, token_type_ids, layers_group_labels + [labels]

        label_ids = torch.zeros(self.n_labels)
        label_ids = label_ids.scatter(0, torch.tensor(labels),
                                      torch.tensor([1.0 for i in labels]))
        return input_ids, attention_mask, token_type_ids, label_ids
    
    def __len__(self):
        return self.len 
