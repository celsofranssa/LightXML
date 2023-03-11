import os
import pickle
import re
import click
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from logzero import logger

from deepxml.data_utils import *


def tokenize(sentence: str, sep='/SEP/'):
    # We added a /SEP/ symbol between titles and descriptions such as Amazon dataset.
    return [token.lower() if token != sep else token for token in word_tokenize(sentence)
            if len(re.sub(r'[^\w]', '', token)) > 0]


def load_ids(dataset, fold_idx, split):
    with open(f"resource/dataset/{dataset}/fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
        return pickle.load(ids_file)


def prepare_data(dataset, fold_idx, split):
    ids = load_ids(dataset, fold_idx, split)
    with open(f"resource/dataset/{dataset}/samples.pkl", "rb") as samples_file:
        samples_df = pd.DataFrame(pickle.load(samples_file))
    samples_df = samples_df[samples_df["idx"].isin(ids)]
    if split == "train":
        checkpoint_sparse_samples(samples_df, dataset)

    checkpoint_samples(samples_df, dataset, split)

def checkpoint_sparse_samples(samples_df, dataset):
    logger.info('Preprocessing sparse features')
    X = TfidfVectorizer().fit_transform(samples_df["text"])
    y = MultiLabelBinarizer(sparse_output=True).fit_transform(samples_df["labels_ids"])
    dump_svmlight_file(X, y=y, f=f"resource/dataset/{dataset}/train_v1.txt", multilabel=True)


def checkpoint_samples(samples_df, dataset, split):
    logger.info('Preprocessing raw texts.')
    dataset_dir = f"resource/dataset/{dataset}/"
    samples_df["text"] = samples_df["text"].apply(lambda text: text.replace("\n", " "))
    samples_df["text"].to_csv(f"{dataset_dir}{split}_raw_texts.txt", header=False, index=False)

    samples_df["labels_ids"] = samples_df["labels_ids"].apply(lambda labels_ids: " ".join([str(idx) for idx in labels_ids]))
    samples_df["labels_ids"].to_csv(f"{dataset_dir}{split}_labels.txt", header=False, index=False)




@click.command()
@click.option('--dataset', help='Name of dataset.')
@click.option('--fold', help='Fold idx.')
@click.option('--text-path', type=click.Path(), help='Path of text.')
@click.option('--tokenized-path', type=click.Path(), default=None, help='Path of tokenized text.')
@click.option('--label-path', type=click.Path(), default=None, help='Path of labels.')
@click.option('--vocab-path', type=click.Path(), default=None,
              help='Path of vocab, if it doesn\'t exit, build one and save it.')
@click.option('--emb-path', type=click.Path(), default=None, help='Path of word embedding.')
@click.option('--w2v-model', type=click.Path(), default=None, help='Path of Gensim Word2Vec Model.')
@click.option('--vocab-size', type=click.INT, default=500000, help='Size of vocab.')
@click.option('--max-len', type=click.INT, default=500, help='Truncated length.')
def main(dataset, fold, text_path, tokenized_path, label_path, vocab_path, emb_path, w2v_model, vocab_size, max_len):
    logger.info('Preparing splits')
    prepare_data(dataset, fold, "train")
    prepare_data(dataset, fold, "test")


    if tokenized_path is not None:
        logger.info(F'Tokenizing Text. {text_path}')
        with open(text_path) as fp, open(tokenized_path, 'w') as fout:
            for line in tqdm(fp, desc='Tokenizing'):
                print(*tokenize(line), file=fout)
        text_path = tokenized_path

    if not os.path.exists(vocab_path):
        logger.info(F'Building Vocab. {text_path}')
        with open(text_path) as fp:
            vocab, emb_init = build_vocab(fp, w2v_model, vocab_size=vocab_size)
        np.save(vocab_path, vocab)
        np.save(emb_path, emb_init)
    vocab = {word: i for i, word in enumerate(np.load(vocab_path, allow_pickle=True))}
    logger.info(F'Vocab Size: {len(vocab)}')

    logger.info(F'Getting Dataset: {text_path} Max Length: {max_len}')
    texts, labels = convert_to_binary(text_path, label_path, max_len, vocab)
    logger.info(F'Size of Samples: {len(texts)}')
    np.save(os.path.splitext(text_path)[0], texts)
    if labels is not None:
        assert len(texts) == len(labels)
        np.save(os.path.splitext(label_path)[0], labels)


if __name__ == '__main__':
    main()
