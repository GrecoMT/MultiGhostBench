from typing import List, Union, Dict

import pandas as pd
import logging

from tqdm import tqdm
from transformers import GPT2TokenizerFast
from functools import lru_cache

logging.basicConfig(level=logging.ERROR)


def load_dataset_valla_no_chunking(dataset_name, unseen_author=None):
    #assert dataset_name.endswith("topic-ood-llm-dataset")
    #assert dataset_name.startswith("config-")

    dataset = pd.read_csv(
            f"./data/dataset/{dataset_name}.csv")

    if unseen_author:
        print("Unseen Author: ", unseen_author)
        dataset = dataset[~(dataset['author'] == unseen_author)]

    authors = sorted(list(set(dataset['author'])))
    logging.info("Authors: " +  str(authors))
    author_id_map = {author: idx for idx, author in enumerate(authors)}

    data = []
    for _, row in dataset.iterrows():
        text = row['text']
        data.append(
            [author_id_map[row['author']], text]
        )
    logging.info(f"Loaded {len(data)} rows.")
    return data, author_id_map


def load_dataset_valla_chunking(dataset_name, unseen_author=None):
    assert dataset_name.endswith("topic-ood-llm-dataset")
    assert dataset_name.startswith("config-")

    dataset = pd.read_csv(
        f"./data/dataset/{dataset_name}.csv")

    if unseen_author:
        print("Unseen Author: ", unseen_author)
        dataset = dataset[~(dataset['author'] == unseen_author)]

    authors = sorted(list(set(dataset['author'])))
    logging.info("Authors: " + str(authors))
    author_id_map = {author: idx for idx, author in enumerate(authors)}

    data = []
    for _, row in dataset.iterrows():
        text = row['text']
        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            data.append(
                [author_id_map[row['author']], chunk]
            )
    logging.info(f"Loaded {len(data)} rows.")
    return data, author_id_map


def load_dataset_ours(filename, split, author_type, test_type=None):
    if split == 'test':
        assert test_type is not None
    elif split == 'train':
        assert test_type is None

    # TODO: remove the below hardcoding
    dataset = pd.read_csv(
        "./data/dataset/" + filename + ".csv")

    dataset = dataset[dataset['split'] == split]
    if test_type is not None:
        dataset = dataset[dataset['type'] == test_type]
    if author_type != 'all':
        dataset = dataset[dataset['author_type'] == author_type]

    return dataset


def load_dataset_mgt(config, test_type, unseen_author):
    train_dataset = pd.read_csv(
        "./data/dataset/" + f"{config}-train-topic-ood-llm-dataset" + ".csv")
    test_dataset = pd.read_csv(
        "./data/dataset/" + f"{config}-test-topic-ood-llm-dataset" + ".csv")
    test_dataset = test_dataset[test_dataset['type'] == test_type]

    if unseen_author:
        print("Unseen Author: ", unseen_author)
        train_dataset = train_dataset[~(train_dataset['author'] == unseen_author)]
        test_dataset = test_dataset[~(test_dataset['author'] == unseen_author)]

    authors = sorted(list(set(train_dataset['author'])))
    author_id_map = {author: idx for idx, author in enumerate(authors)}

    print("Authors: " + str(authors))

    data = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        }

    }

    for _, row in tqdm(test_dataset.iterrows(), total=len(test_dataset)):
        text = row['text']
        data['test']['text'].append(text)
        data['test']['label'].append(author_id_map[row['author']])
    assert len(data['test']['text']) == len(data['test']['label'])

    for _, row in tqdm(train_dataset.iterrows(), total=len(train_dataset)):
        text = row['text']
        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            data['train']['text'].append(chunk)
            data['train']['label'].append(author_id_map[row['author']])
    assert len(data['train']['text']) == len(data['train']['label'])

    print(f"Loaded for {config} for test type: {test_type}; Has {len(data['train']['text'])} train rows "
                 f"and {len(data['test']['text'])} test rows.")
    return data, author_id_map


@lru_cache(maxsize=1)
def get_tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


def split_text_into_chunks(text: str, chunk_size: int = 512):
    tokens = get_tokenizer().encode(text)

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [get_tokenizer().decode(chunk) for chunk in chunks]

    return text_chunks


def aa_as_pandas(data: List[List[Union[int, str]]]) -> pd.DataFrame:
    return pd.DataFrame(data, columns=['labels', 'text'])


def list_dset_to_dict(data: List[List[Union[int, str]]]) -> Dict:
    out = {}
    for auth, text in data:
        out.setdefault(auth, []).append(text)
    return out


def dict_dset_to_list(data: Dict) -> List[List[Union[int, str]]]:
    out = []
    for auth, texts in data.items():
        for text in texts:
            out.append([auth, text])
    return out
