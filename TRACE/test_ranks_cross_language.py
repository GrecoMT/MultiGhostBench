import argparse
from pprint import pformat

import pandas as pd
import pickle
import logging
import wandb
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, accuracy_score
import torch
import wandb
from datasets import tqdm
from scipy.spatial.distance import jensenshannon
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)

def get_wandb_run_name(args):
    return f"our-ranks-cluster-{args.cluster}-metric-{args.metric}-model-{args.model}-context-{args.context_size}-alpha-{args.alpha}-{args.resource}-train-{args.language_train}-test-{args.language_test}-llm-dataset-threshold-{args.threshold}"


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
            'recall': class_recall
        })
    # macro_f1 = np.mean(class_f1s) if class_f1s else 0
    return pd.DataFrame(results)

def get_final_cluster_num(rank_to_cluster_num_map):
    return list(rank_to_cluster_num_map.values())[-1] + 1


def get_fingerprint(ranks, rank_to_cluster_num_map):
    final_cluster_num = get_final_cluster_num(rank_to_cluster_num_map)
    fingerprint = np.zeros((final_cluster_num, final_cluster_num), dtype=np.int32)
    prev_rank = ranks[0]
    for curr_rank in (ranks[1:]):
        curr_row_cluster = rank_to_cluster_num_map[prev_rank]
        curr_column_cluster = rank_to_cluster_num_map[curr_rank]
        fingerprint[curr_row_cluster, curr_column_cluster] += 1
        prev_rank = curr_rank
    return fingerprint


def get_ref_fingerprint(args, method, author, data, rank_to_cluster_num_map):
    final_cluster_num = get_final_cluster_num(rank_to_cluster_num_map)
    fingerprints = []
    author_data = data[data['author'] == author]
    for _, row in author_data.iterrows():
        fingerprint = np.zeros((final_cluster_num, final_cluster_num), dtype=np.int32)
        curr_novel = row['novel_name']
        ranks = np.load(
            #f"./data/{method}/{args.model}-context-{args.context_size}-llm-novel-{curr_novel}.txt-{method}.npz")['ranks']
            f"./data/TRACE-scores/{args.language_train}/{method}/{args.model}-context-{args.context_size}-llm-novel-{curr_novel}.txt-{method}.npz")['ranks']
        prev_rank = ranks[0]
        for curr_rank in (ranks[1:]):
            curr_row_cluster = rank_to_cluster_num_map[prev_rank]
            curr_column_cluster = rank_to_cluster_num_map[curr_rank]
            fingerprint[curr_row_cluster, curr_column_cluster] += 1
            prev_rank = curr_rank
        fingerprints.append(fingerprint)
    return fingerprints


def get_score(metric, fingerprint1, fingerprint2):
    if metric == 'cos_sim':
        return cosine_similarity([fingerprint1.flatten()], [fingerprint2.flatten()])[0, 0]
    elif metric == 'frob_norm':
        return -1 * np.linalg.norm(fingerprint1 - fingerprint2, ord=2)
    elif metric == 'wass_dist':
        raise NotImplementedError
    elif metric == 'js_dist':
        return 1 - jensenshannon(fingerprint1.flatten(), fingerprint2.flatten())
    else:
        raise NotImplementedError("Not supported metric")


def get_predicted_author(ranks, author_ref_fingerprints_map, metric, rank_to_cluster_num_map):
    curr_fingerprint = get_fingerprint(ranks, rank_to_cluster_num_map)
    scores = []
    for author, author_fingerprints in author_ref_fingerprints_map.items():
        scores.append(np.max([get_score(metric, author_fingerprint, curr_fingerprint) for author_fingerprint in author_fingerprints]))
    return np.argmax(scores), np.array(scores)


def get_ranks_predictions(args, test_data, author_ref_fingerprints_map, rank_to_cluster_num_map):
    y_pred = []
    y_scores = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        curr_novel = row['novel_name']
        ranks = np.load(
            f"./data/TRACE-scores/{args.language_test}/ranks/{args.model}-context-{args.context_size}-llm-novel-{curr_novel}.txt-ranks.npz")['ranks']
        pred_author, score = get_predicted_author(ranks, author_ref_fingerprints_map, args.metric, rank_to_cluster_num_map)
        y_pred.append(pred_author)
        y_scores.append(score)
    return np.array(y_pred), np.array(y_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Trace-Rank model from the command line')
    #parser.add_argument('--config', type=str)

    parser.add_argument('--resource', type=str)
    parser.add_argument('--language_train', type=str)
    parser.add_argument('--language_test', type=str)

    parser.add_argument('--threshold', type=str)
    parser.add_argument('--cluster', type=str)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--model', type=str)
    #parser.add_argument('--context', type=str)
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--context_size', type=str)
    args = parser.parse_args()
    args.threshold = float(args.threshold)
    args.cluster = int(args.cluster)
    args.alpha = float(args.alpha)
    args.context_size = int(args.context_size)
    logging.info("Args:\n%s", pformat(args))
    wandb.login()
    wandb.init(project="multilingual-OOD-AA", name=get_wandb_run_name(args), config=args)
    THRESHOLD = args.threshold
    
    dataset = pd.read_csv(f"./data/dataset-{args.resource}-res/train-{args.language_train}-llm-dataset.csv") ##only for labels

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



    train_data = pd.read_csv(f"./data/dataset-{args.resource}-res/train-{args.language_train}-llm-dataset.csv")

    rank_to_cluster_num_map = pd.read_pickle(f'./rank-compression/data/{args.model}-100k-ranks-power-law-alpha-{args.alpha}-cluster-map-{args.cluster}.pkl')
    author_ref_fingerprints_map = {}
    for author in authors:
        author_ref_fingerprints_map[author] = get_ref_fingerprint(args, "ranks", author, train_data,
                                                                  rank_to_cluster_num_map)

    ID_test_predictions, ID_test_probs = get_ranks_predictions(args, ID_test_data, author_ref_fingerprints_map, rank_to_cluster_num_map)
    pd.DataFrame({
        'ID_test_predictions': ID_test_predictions,
        'ID_test_labels': ID_test_labels,
    }).to_csv(f"./data/results/{get_wandb_run_name(args)}-ID_predictions.csv", index=False)

    no_threshold_ID_results = get_results(ID_test_probs, ID_test_labels, 0, author_id_map, id_author_map)
    print("\nNO THRESHOLD:")
    print("ID:")

    print("all: ", np.mean(no_threshold_ID_results['f1']))

    no_threshold_result = {
            "no_threshold/ID/all": np.mean(no_threshold_ID_results['f1'])
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
            
            "threshold/ID/all": np.mean(ID_results['f1']),
    }
    wandb.log(threshold_result)
    wandb.run.summary.update(threshold_result)

    wandb.finish()
