import os
import random
import re
import urllib.request
import zipfile
from collections import Counter

import numpy as np

DATA_DIR = "./data"
TINY_SHAKESPEARE_FILENAME = "tinyshakespeare.txt"
TEXT8_CORPUS_FILENAME = "text8.txt"
WIKITEXT2_FILENAME = "wikitext2.txt"

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TEXT8_CORPUS_URL = "http://mattmahoney.net/dc/text8.zip"
WIKITEXT2_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"

UNK_TOKEN = "<UNK>"

DATASETS = {
    "tiny-shakespeare": {
        "url": TINY_SHAKESPEARE_URL,
        "filename": TINY_SHAKESPEARE_FILENAME,
    },
    "text8-corpus": {
        "url": TEXT8_CORPUS_URL,
        "filename": TEXT8_CORPUS_FILENAME,
    },
    "wikitext-2": {
        "url": WIKITEXT2_URL,
        "filename": WIKITEXT2_FILENAME,
    },
}


def download_data(dataset_name):
    target_filename = DATASETS[dataset_name]["filename"]
    filepath = os.path.join(DATA_DIR, target_filename)

    # Check if the file already exists
    if os.path.exists(filepath):
        print(f"{dataset_name} already downloaded")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    url = DATASETS[dataset_name]["url"]

    if url.endswith(".zip"):
        zip_path = os.path.join(DATA_DIR, "temp.zip")
        print(f"Downloading {dataset_name} zip archive...")
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

        # The extracted file inside text8.zip is named 'text8'
        extracted_file = os.path.join(DATA_DIR, "text8")
        if os.path.exists(extracted_file):
            os.rename(extracted_file, filepath)

        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"Extracted to {filepath}")
    else:
        print(f"Downloading {dataset_name}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")


def load_data(dataset_name):
    dataset = None

    # load the data:
    with open(
        os.path.join(DATA_DIR, DATASETS[dataset_name]["filename"]),
        "r",
        encoding="utf-8",
    ) as f:
        dataset = f.read()

    return dataset


def build_vocab(text, min_freq=5):
    # Step 1: Clean and tokenize raw text into a list of word strings
    words = re.findall(r"\w+", text.lower())

    # Step 2: Count frequencies of every word
    word_counts = Counter(words)

    # Step 3: filter if the words do not meet the min_feq
    vocab = [word for word, counts in word_counts.items() if counts > min_freq]

    # Step 4: add the unknown token for words that are not present in the vocab
    vocab.append(UNK_TOKEN)

    # Step 5: build the word2id and id2word dicts
    word2id = {word: idx for idx, word in enumerate(vocab)}
    id2word = {idx: word for idx, word in enumerate(vocab)}

    return word2id, id2word, words


def subsample_tokens(words, threshold=1e-4):
    word_counts = Counter(words)

    keep_probs = {
        word: min(1.0, (threshold / (counts / len(words))) ** 0.5)
        for word, counts in word_counts.items()
    }

    return [word for word in words if random.random() < keep_probs[word]]


def build_noise_table(words, word2id, table_size=1_000_000):
    word_counts = Counter(words)

    vocab_words = [w for w in word2id.keys() if w != UNK_TOKEN]
    vocab_ids = np.array([word2id[w] for w in vocab_words])

    pow_counts = np.array([word_counts[w] for w in word2id.keys() if w != UNK_TOKEN]) ** 0.75
    probs = pow_counts / np.sum(pow_counts)

    noise_table = np.random.choice(vocab_ids, size=table_size, replace=True, p=probs)
    return noise_table


def generate_cbow_pairs(tokens, window_size=2):
    pairs = []
    for i in range(window_size, len(tokens) - window_size):
        target_id = tokens[i]
        context_ids = tokens[i - window_size : i + window_size + 1]
        context_ids.pop(window_size)

        pairs.append([context_ids, target_id])
    return pairs


def generate_skip_gram_pairs(tokens, window_size=2):
    pairs = []
    for i in range(window_size, len(tokens) - window_size):
        target_id = tokens[i]
        context_ids = tokens[i - window_size : i + window_size + 1]
        context_ids.pop(window_size)

        pairs.extend([[target_id, context_id] for context_id in context_ids])
    return pairs


def generate_batches(pairs, batch_size=32):
    batches = []
    for i in range(0, len(pairs), batch_size):
        batches.append(pairs[i : i + batch_size])

    return batches


if __name__ == "__main__":
    download_data("wikitext-2")
    dataset = load_data("wikitext-2")

    word2id, id2word, words = build_vocab(dataset)
    noise_table = build_noise_table(words, word2id)
    print(noise_table)
