import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import pickle
import os
import pickle
import re
import argparse
import time
import logging
from pprint import pformat
from collections import Counter
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.text_embedding import TextEmbeddingModel
from src.index import Indexer

import numpy as np
import wandb
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from typing import List, Callable, Tuple


logging.basicConfig(level=logging.ERROR)

def get_wandb_run_name(args):
    #return f"Detective-{args.config}-train-topic-ood-llm-dataset-threshold-{args.threshold}"
    return f"Detective-{args.resource}-{args.language_train}-train-test-{args.language_test}-llm-dataset-threshold-{args.threshold}"
    

from functools import lru_cache
#from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer
@lru_cache(maxsize=1)
def get_tokenizer():
    #return GPT2TokenizerFast.from_pretrained("gpt2")
    return AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

def split_text_into_chunks(text: str, chunk_size: int = 512):
    tokens = get_tokenizer().encode(text)

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [get_tokenizer().decode(chunk) for chunk in chunks]

    return text_chunks


def process_top_ids_and_scores_AA(top_ids_and_scores, label_dict):
    preds = []
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        num_dict = {}
        max_num, max_id = 0, 0
        for id in ids:
            if label_dict[int(id)] not in num_dict:
                num_dict[label_dict[int(id)]] = 1
            else:
                num_dict[label_dict[int(id)]] += 1
            if num_dict[label_dict[int(id)]] > max_num:
                max_num = num_dict[label_dict[int(id)]]
                max_id = label_dict[int(id)]
        preds.append(str(max_id))
    return preds


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_detective_prediction_per_text(model, tokenizer, index, chunks, label_dict, batch_size=128, show_progress=False):
    model.eval()

    all_embeddings = []

    # Create DataLoader for automatic batching & memory management
    loader = DataLoader(
        chunks,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # tune or set 0 on windows if issues
        collate_fn=lambda x: x  # identity -> we encode inside loop
    )

    progress = tqdm(loader, disable=not show_progress, desc="Encoding")

    with torch.no_grad():
        for batch_texts in progress:
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True,
            )

            encoded = {k: v.to("cuda", non_blocking=True)
                       for k, v in encoded.items()}

            emb = model(encoded)

            # Move to cpu & numpy in one go
            all_embeddings.append(emb.cpu().numpy())

    # Final concatenation - this is the only big allocation on CPU RAM
    embeddings = np.concatenate(all_embeddings, axis=0)

    # Now search (hopefully your index is on CPU or can handle big queries)
    top_ids_and_scores = index.search_knn(embeddings, K)

    preds = [int(pred) for pred in process_top_ids_and_scores_AA(top_ids_and_scores, label_dict)]
    # print(preds)
    total = len(preds)
    counts = Counter(preds)
    # print (counts)

    # Most frequent class (highest count wins, ties → lowest ID wins)
    majority_class = max(range(len(author_id_map)), key=lambda i: counts.get(i, 0))
    # print (majority_class)

    percentages = []
    for i in range(len(author_id_map)):
        count = counts.get(i, 0)
        percentage = (count / total)
        percentages.append(percentage)

    return majority_class, percentages


def get_detective_predictions(model, tokenizer, index, texts, label_dict):
    predictions = []
    raw_outputs = []
    for text in tqdm(texts, desc="Predicting for long texts"):
        chunks = split_text_into_chunks(text)
        # print("chunks: ", len(chunks))
        majority_class, percentages = get_detective_prediction_per_text(model, tokenizer, index, chunks, label_dict)
        assert majority_class == np.argmax(percentages), f"{majority_class}, {percentages}"
        predictions.append(majority_class)
        raw_outputs.append(percentages)
    return predictions, raw_outputs


def get_results(y_pred_probs, y_true, threshold, author_id_map, id_author_map):
    results = []
    assert len(y_pred_probs) == len(y_true)
    top1_probs = np.max(y_pred_probs, axis=1)
    top1_classes = np.argmax(y_pred_probs, axis=1)
    accepted_mask = top1_probs >= threshold
    class_f1s = []
    class_accuracies = []
    class_precisions = []
    class_recalls = []
    n_classes = len(author_id_map)
    for class_id in range(n_classes):
        # Samples belonging to this class
        class_mask = (y_true == class_id)

        if np.sum(class_mask) == 0:
            # print(threshold, class_id, "empty")
            continue  # Skip classes with no samples

        # Among samples of this class, which were accepted?
        class_predicted_correct = (top1_classes == class_id) & class_mask & accepted_mask

        # Class recall: accepted and correct / total in class
        class_recall = np.sum(class_predicted_correct) / np.sum(class_mask)

        # Class precision: among all predictions for this class (accepted), how many correct?
        predicted_as_class = (top1_classes == class_id) & accepted_mask
        if np.sum(predicted_as_class) > 0:
            class_precision = np.sum(class_predicted_correct) / np.sum(predicted_as_class)
        else:
            class_precision = 0

        # Class accuracy: correct predictions / total in class
        class_accuracy = np.sum(class_predicted_correct) / np.sum(class_mask)

        # Class F1
        if class_precision + class_recall > 0:
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
        else:
            class_f1 = 0

        class_accuracies.append(class_accuracy)
        class_precisions.append(class_precision)
        class_recalls.append(class_recall)
        class_f1s.append(class_f1)
        author = id_author_map[class_id]
        results.append({
            'author': author,
            'label_id': class_id,
            'f1': class_f1,
            'acc': class_accuracy,
            'precision': class_precision,
            'recall': class_recall,
        })
    # macro_f1 = np.mean(class_f1s) if class_f1s else 0
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a detective model from the command line')
    #parser.add_argument('--config', type=str)
    
    parser.add_argument('--language_train', type=str)
    parser.add_argument('--language_test', type=str)
    parser.add_argument('--resource', type=str)
    
    parser.add_argument('--threshold', type=str)
    args = parser.parse_args()
    args.threshold = float(args.threshold)
    logging.info("Args:\n%s", pformat(args))
    wandb.login()
    wandb.init(project="multilingual-OOD-AA", name=get_wandb_run_name(args), config=args)
    THRESHOLD = args.threshold
    
    dataset = pd.read_csv(
        f"../data/dataset-{args.resource}-res/train-{args.language_train}-llm-dataset.csv") #for labels

    authors = sorted(list(set(dataset['author'])))
    logging.info(f'Number of authors: {len(authors)}')
    logging.info(f'Authors: {authors}')
    author_id_map = {author: idx for idx, author in enumerate(authors)}
    id_author_map = {idx: author for author, idx in author_id_map.items()}

    test_data = pd.read_csv(f"../data/dataset-{args.resource}-res/test-{args.language_test}-llm-dataset.csv")
    ID_test_data = test_data[test_data['type'] == 'ID']
    ID_test_texts = list(ID_test_data['text'])
    ID_test_labels = np.array([author_id_map[author] for author in list(ID_test_data['author'])])
    assert len(ID_test_texts) == len(ID_test_labels)


    model_name = "ZurichNLP/unsup-simcse-xlm-roberta-base"
    #model_path = f"../DeTeCtive/src/runs/{args.resource}-{args.language_train}-OOD-unseen-None_v0/model_best.pth"
    model_path = f"../DeTeCtive/src/runstry /{args.resource}-{args.language_train}-OOD-unseen-None_v0/model_best.pth"
    embedding_dim = 768
    database_path = f"../DeTeCtive/src/database/database-{args.resource}-{args.language_train}-unseen-None"
    K = 5

    model = TextEmbeddingModel(model_name).cuda()
    state_dict = torch.load(model_path, map_location=model.model.device)
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = state_dict[key]
    model.load_state_dict(state_dict)
    tokenizer = model.tokenizer

    index = Indexer(embedding_dim)
    index.deserialize_from(database_path)
    label_dict = load_pkl(os.path.join(database_path, 'label_dict.pkl'))

    ID_test_predictions, ID_test_probs = get_detective_predictions(model, tokenizer, index, ID_test_texts, label_dict)

    pd.DataFrame({
        'ID_test_predictions': ID_test_predictions,
        'ID_test_labels': ID_test_labels,
    }).to_csv(f"../data/results/{get_wandb_run_name(args)}-ID_predictions.csv", index=False)


    no_threshold_ID_results = get_results(ID_test_probs, ID_test_labels, 0, author_id_map, id_author_map)
    print("\nNO THRESHOLD:")
    print("ID:")
    print("all: ", np.mean(no_threshold_ID_results['f1']))


    no_threshold_result = {
            "no_threshold/ID/all": np.mean(no_threshold_ID_results['f1']),
    }
    no_threshold_ID_results.to_csv(f"../data/results/{get_wandb_run_name(args)}-no_threshold_ID_results.csv", index=False)

    wandb.log(no_threshold_result)
    wandb.run.summary.update(no_threshold_result)

    print("\nTHRESHOLD:", THRESHOLD)
   
    ID_results = get_results(ID_test_probs, ID_test_labels, THRESHOLD, author_id_map, id_author_map)
    print("ID:")
    print("all: ", np.mean(ID_results['f1']))

   

    ID_results.to_csv(f"../data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_ID_results.csv",
                                   index=False)

    threshold_result = {
            "threshold/ID/all": np.mean(ID_results['f1']),
    }
    wandb.log(threshold_result)
    wandb.run.summary.update(threshold_result)

    wandb.finish()
