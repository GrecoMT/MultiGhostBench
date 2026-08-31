from functools import lru_cache

import pandas as pd
import random
import tqdm
import re
import numpy as np
import os
import json

#from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import AutoTokenizer



def trim_quotes(s):
    return s.strip("\"'")

def process_spaces(text):
    return text
    # text=text.replace(
    #     ' ,', ',').replace(
    #     ' .', '.').replace(
    #     ' ?', '?').replace(
    #     ' !', '!').replace(
    #     ' ;', ';').replace(
    #     ' \'', '\'').replace(
    #     ' ’ ', '\'').replace(
    #     ' :', ':').replace(
    #     '<newline>', '\n').replace(
    #     '`` ', '"').replace(
    #     ' \'\'', '"').replace(
    #     '\'\'', '"').replace(
    #     '.. ', '... ').replace(
    #     ' )', ')').replace(
    #     '( ', '(').replace(
    #     ' n\'t', 'n\'t').replace(
    #     ' i ', ' I ').replace(
    #     ' i\'', ' I\'').replace(
    #     '\\\'', '\'').replace(
    #     '\n ', '\n').strip()
    # return trim_quotes(text)


@lru_cache(maxsize=1)
def get_tokenizer():
    #return GPT2TokenizerFast.from_pretrained("gpt2")
    return AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")


def split_text_into_chunks(text: str, chunk_size: int = 512):
    tokens = get_tokenizer().encode(text)

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [get_tokenizer().decode(chunk) for chunk in chunks]

    return text_chunks


#def load_OOD(config, unseen_author):
def load_OOD(resource, language, unseen_author):
    data={
        'train':[],
        'test':[],
        'valid':[]
    }
    #print(f"Loading OOD AA Data for {config} and unseen author: {unseen_author}")
    print(f"Loading OOD AA Data for {resource}-resource-{language} and unseen author: {unseen_author}")

    #train_data = pd.read_csv(f"/data/gpfs/projects/punim2157/code/AA/LLM-OOD-AA/data/dataset/{config}-train-topic-ood-llm-dataset.csv")
    train_data = pd.read_csv(f"/data/projects/punim0478/matteogreco/multilingual-OOD-AA/data/dataset-{resource}-res/train-{language}-llm-dataset.csv")
    #test_data = pd.read_csv(f"/data/gpfs/projects/punim2157/code/AA/LLM-OOD-AA/data/dataset/{config}-test-topic-ood-llm-dataset.csv")
    test_data = pd.read_csv(f"/data/projects/punim0478/matteogreco/multilingual-OOD-AA/data/dataset-{resource}-res/test-{language}-llm-dataset.csv")
    #valid_data = pd.read_csv(f"/data/gpfs/projects/punim2157/code/AA/LLM-OOD-AA/data/dataset/dev-time-ood-llm-dataset.csv")
    valid_data = pd.read_csv(f"/data/projects/punim0478/matteogreco/multilingual-OOD-AA/data/dataset-{resource}-res/devset-{language}.csv")

    train_data = train_data.rename(columns={
        'text': 'Generation',
        'author': 'label'
    })

    valid_data = valid_data.rename(columns={
        'text': 'Generation',
        'author': 'label'
    })

    test_data = test_data.rename(columns={
        'text': 'Generation',
        'author': 'label'
    })

    train_data = train_data[['Generation', 'label']]
    valid_data = valid_data[['Generation', 'label']]
    test_data = test_data[['Generation', 'label']]

    all_authors = sorted(
    set(train_data["label"]) |
    set(valid_data["label"]) |
    set(test_data["label"])
    )

    author_to_id = {author: i for i, author in enumerate(all_authors)}

    for i in range(len(train_data)):
        text=train_data.iloc[i]['Generation']
        src = train_data.iloc[i]['label']
        if src == unseen_author:
            continue
        for chunk in split_text_into_chunks(text):
            label= '-1'  # TODO: not sure what is the use of this and impact
            data["train"].append((process_spaces(str(chunk)),label,src,i))

    for i in range(len(valid_data)):
        text = valid_data.iloc[i]['Generation']
        src = valid_data.iloc[i]['label']
        if src == unseen_author:
            continue
        for chunk in split_text_into_chunks(text):
            label = '-1'  # TODO: not sure what is the use of this and impact
            data["valid"].append((process_spaces(str(chunk)), label, src, i))
    for i in range(len(test_data)):
        text = test_data.iloc[i]['Generation']
        src = test_data.iloc[i]['label']
        if src == unseen_author:
            continue
        label = '-1'  # TODO: not sure what is the use of this and impact
        data["test"].append((process_spaces(str(text)), label, src, i))
    
    print (f"loaded OOD data, train:{len(data['train'])}, test:{len(data['test'])}, valid:{len(data['valid'])}")
    return data
