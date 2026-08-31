import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

import numpy as np
import wandb
from simpletransformers.classification import ClassificationModel
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from typing import List, Callable, Tuple


logging.basicConfig(level=logging.INFO)


def get_wandb_run_name(args):
    return f"BertAA-train-{args.language_train}-test-{args.language_test}-llm-dataset-threshold-{args.threshold}"

from functools import lru_cache
from transformers import AutoTokenizer

@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')

def split_text_into_chunks(text: str, chunk_size: int = 512):
    tokens = get_tokenizer().encode(text)

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [get_tokenizer().decode(chunk) for chunk in chunks]

    return text_chunks

#from datasets import tqdm
from tqdm import tqdm
from transformers import BertForSequenceClassification, XLMRobertaTokenizer
from scipy.special import softmax


# Optional: Move to GPU if available
import torch

def get_bertaa_predictions(args, texts):
    # Replace with the exact directory path where it was saved (e.g., "./BertAA/your-run-name")
    # BertAA-llm-train-topic-ood-dataset-unseen-None-epochs-5
    
    #saved_dir = f"./outputs-bertAA/outputs-{args.resource}res-BertAA/BertAA-train-{args.language_train}-llm-dataset-unseen-None-epochs-2"  # Use the actual folder name created during saving

    saved_dir = f"./outputs-bertAA/outputs-{args.resource}res-BertAA/{args.language_train}/BertAA-train-{args.language_train}-llm-dataset-unseen-None-epochs-{args.epochs}"  # Use the actual folder name created during saving
    
    model = ClassificationModel(
        "xlmroberta", # or your model_type, e.g., "roberta"s
        saved_dir,   # path to the saved directory
        num_labels=5,   # must match training
        # args={                   # use the SAME args as training (important!)
        #     # Include relevant training_args here, e.g.:
        #     # "overwrite_output_dir": True,
        #     # "no_cuda": params.no_cuda,
        #     # etc.
        # },
        use_cuda=True,
        cuda_device=0
    )
    
    #print(model.model)
    
    predictions = []
    raw_outputs = []

    for text in tqdm(texts, total=len(texts), desc="Predicting for test texts"):
        chunks = split_text_into_chunks(text)

        chunk_predictions, chunk_raw_outputs = model.predict(chunks)
        # chunk_probs = [softmax(logits) for logits in chunk_raw_outputs]
        chunk_probs = [softmax(logits) for logits in chunk_raw_outputs]

        # majority_pred = Counter(chunk_predictions).most_common(1)[0][0]
        avg_probs = np.mean(chunk_probs, axis=0)

        predictions.append(np.argmax(avg_probs))
        raw_outputs.append(avg_probs)

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
    parser = argparse.ArgumentParser(description='Run a N-Gram model from the command line')
    #parser.add_argument('--config', type=str)

    parser.add_argument('--language_train', type=str) 
    parser.add_argument('--language_test', type=str) 
    parser.add_argument('--resource', type=str)
    parser.add_argument('--epochs', default=2)

    parser.add_argument('--threshold', type=str)
    args = parser.parse_args()
    args.threshold = float(args.threshold)
    logging.info("Args:\n%s", pformat(args))
    wandb.login()
    wandb.init(project="multilingual-OOD-AA", name=get_wandb_run_name(args), config=args)
    THRESHOLD = args.threshold

    dataset = pd.read_csv(
        f"./data/dataset-{args.resource}-res/train-{args.language_train}-llm-dataset.csv")

    authors = sorted(list(set(dataset['author'])))
    logging.info(f'Number of authors: {len(authors)}')
    logging.info(f'Authors: {authors}')
    author_id_map = {author: idx for idx, author in enumerate(authors)}
    id_author_map = {idx: author for author, idx in author_id_map.items()}

    test_data = pd.read_csv(f"./data/dataset-{args.resource}-res/test-{args.language_test}-llm-dataset.csv")
    ID_test_data = test_data[test_data['type'] == 'ID']
    ID_test_texts = list(ID_test_data['text'])
    ID_test_labels = np.array([author_id_map[author] for author in list(ID_test_data['author'])])
    assert len(ID_test_texts) == len(ID_test_labels)


    ID_test_predictions, ID_test_probs = get_bertaa_predictions(args, ID_test_texts)

    pd.DataFrame({
        'ID_test_predictions': ID_test_predictions,
        'ID_test_labels': ID_test_labels,
    }).to_csv(f"./data/results/{get_wandb_run_name(args)}-ID_predictions.csv", index=False)

    no_threshold_ID_results = get_results(ID_test_probs, ID_test_labels, 0, author_id_map, id_author_map)
    print("\nNO THRESHOLD:")
    print("ID:")
    print("all: ", np.mean(no_threshold_ID_results['f1']))

    no_threshold_result = {
            "no_threshold/ID/all": np.mean(no_threshold_ID_results['f1']),
    }
    no_threshold_ID_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-no_threshold_ID_results.csv", index=False)
    wandb.log(no_threshold_result)
    wandb.run.summary.update(no_threshold_result)

    print("\nTHRESHOLD:", THRESHOLD)

    ID_results = get_results(ID_test_probs, ID_test_labels, THRESHOLD, author_id_map, id_author_map)
    print("ID:")

    print("all: ", np.mean(ID_results['f1']))


    ID_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_ID_results.csv",
                                   index=False)

    threshold_result = {
            "threshold/ID/all": np.mean(ID_results['f1'])
    }
    wandb.log(threshold_result)
    wandb.run.summary.update(threshold_result)

    wandb.finish()
