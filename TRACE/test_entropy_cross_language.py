import argparse
from pprint import pformat

import pandas as pd
import pickle
import logging

from scipy.stats import gaussian_kde

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
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

def get_wandb_run_name(args):
    return f"our-entropy-grid-{args.grid}-metric-{args.metric}-model-{args.model}-context-{args.context_size}-{args.resource}-res-train-{args.language_train}-test-{args.language_test}-llm-dataset-threshold-{args.threshold}"

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


def get_unseen_results(y_pred_probs, threshold, unseen_author):
    # unseen author case - model says I don't know
    top1_probs = np.max(y_pred_probs, axis=1)
    print(top1_probs)
    accepted_mask = top1_probs < threshold
    y_pred = [int(curr) for curr in accepted_mask]
    y_true = [1] * len(top1_probs)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'unseen_author': unseen_author,
        'f1': f1,
        'acc': acc,
        'precision': prec,
        'recall': rec,
        'samples': len(top1_probs),
    }


def create_signature(data, grid_size, max_value):
    kde = gaussian_kde(data)

    x_grid = np.linspace(0, max_value, grid_size)
    y_grid = np.linspace(0, max_value, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = kde(positions).reshape(X.shape)
    return X, Y, Z


def get_fingerprint(values, grid_size, max_value):
    pairs = [(values[i - 1], values[i]) for i in range(1, len(values))]
    pair_x = [pair[0] for pair in pairs]
    pair_y = [pair[1] for pair in pairs]
    data = np.vstack([pair_x, pair_y])
    X, Y, Z = create_signature(data, grid_size, max_value)
    return Z


def get_ref_fingerprint(args, method, author, data, max_value):
    author_data = data[data['author'] == author]
    print("Author: {}; {}".format(author, len(author_data)))
    fingerprints = []

    for _, row in author_data.iterrows():
        curr_novel = row['novel_name']
        entropies = np.load(
            f"./data/TRACE-scores/{args.language_train}/{method}/{args.model}-context-{args.context_size}-llm-novel-{curr_novel}.txt-{method}.npz")[method]
        pairs = [(entropies[i - 1], entropies[i]) for i in range(1, len(entropies))]
        pair_x = [pair[0] for pair in pairs]
        pair_y = [pair[1] for pair in pairs]
        data = np.vstack([pair_x, pair_y])
        X, Y, Z = create_signature(data, args.grid, max_value)
        fingerprints.append(Z)
    return fingerprints


def compute_normals(Z, dx, dy):
    # Compute gradients using central differences
    Zx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2 * dx)
    Zy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2 * dy)
    # Normal vector: (-Zx, -Zy, 1)
    normals = np.stack([-Zx, -Zy, np.ones_like(Z)], axis=-1)
    # Normalize
    norm = np.sqrt(normals[..., 0] ** 2 + normals[..., 1] ** 2 + normals[..., 2] ** 2)
    normals = normals / norm[..., np.newaxis]
    return normals


def get_mean_normal_angle(Z1, Z2, grid_size, max_value):
    x = np.linspace(0, max_value, grid_size)
    y = np.linspace(0, max_value, grid_size)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # Compute normals
    normals1 = compute_normals(Z1, dx, dy)
    normals2 = compute_normals(Z2, dx, dy)

    # Compute angle between normals (in degrees)
    dot_product = np.sum(normals1 * normals2, axis=-1)
    dot_product = np.clip(dot_product, -1, 1)  # Avoid numerical errors
    angle_diff = np.arccos(dot_product) * 180 / np.pi

    # Normal comparison metrics
    mean_angle = np.mean(angle_diff)
    return mean_angle * 100


def get_score(metric, fingerprint1, fingerprint2, grid, max_value):
    if metric == 'cos_sim':
        return cosine_similarity([fingerprint1.flatten()], [fingerprint2.flatten()])[0, 0]
    elif metric == 'frob_norm':
        return -1 * np.linalg.norm(fingerprint1 - fingerprint2, ord=2)
    elif metric == 'wass_dist':
        raise NotImplementedError
    elif metric == 'js_dist':
        return 1 - jensenshannon(fingerprint1.flatten(), fingerprint2.flatten())
    elif metric == "ssim":
        raise NotImplementedError
    elif metric == 'norm_mean':
        Z1 = fingerprint1
        Z2 = fingerprint2
        return -1 * get_mean_normal_angle(Z1, Z2, grid, max_value)

def get_predicted_author(args, entropies, author_ref_fingerprints_map, metric, max_value):
    curr_fingerprint = get_fingerprint(entropies, args.grid, max_value)
    scores = []
    for author, author_fingerprints in author_ref_fingerprints_map.items():
        scores.append(np.max([get_score(metric, author_fingerprint, curr_fingerprint, args.grid, max_value) for author_fingerprint in author_fingerprints]))
    return np.argmax(scores), np.array(scores)


def get_entropy_predictions(args, test_data, author_ref_fingerprints_map, max_value):
    y_pred = []
    y_scores = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        curr_novel = row['novel_name']
        entropy = np.load(
            #f"./data/entropy/{args.model}-context-{args.context_size}-llm-novel-{curr_novel}.txt-entropy.npz")['entropy']
            f"./data/TRACE-scores/{args.language_test}/entropy/{args.model}-context-{args.context_size}-llm-novel-{curr_novel}.txt-entropy.npz")['entropy']
        pred_author, score = get_predicted_author(args, entropy, author_ref_fingerprints_map, args.metric, max_value)
        y_pred.append(pred_author)
        y_scores.append(score)
    return np.array(y_pred), np.array(y_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Trace-Entropy model from the command line')
    
    parser.add_argument('--resource', type=str)
    parser.add_argument('--language_train', type=str)
    parser.add_argument('--language_test', type=str)

    parser.add_argument('--threshold', type=str)
    parser.add_argument('--grid', type=str)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--context_size', type=str)
    args = parser.parse_args()
    args.threshold = float(args.threshold)
    args.grid = int(args.grid)
    args.context_size = int(args.context_size)
    logging.info("Args:\n%s", pformat(args))
    wandb.login()
    wandb.init(project="multilingual-OOD-AA", name=get_wandb_run_name(args), config=args)
    THRESHOLD = args.threshold

    tokenizer = AutoTokenizer.from_pretrained(args.model.replace("-", "/", 1))
    vocab_size = tokenizer.vocab_size
    max_value = np.log2(vocab_size)
    print("vocab size: ", vocab_size, "max_value: ", max_value)
    
    dataset = pd.read_csv(f"./data/dataset-{args.resource}-res/train-{args.language_train}-llm-dataset.csv") ##for labels

    NO_THRESHOLD = 0
    if args.metric == "norm_mean":
        NO_THRESHOLD = -1 * 100 * 180

    authors = sorted(list(set(dataset['author'])))
    logging.info(f'Number of authors: {len(authors)}')
    logging.info(f'Authors: {authors}')
    author_id_map = {author: idx for idx, author in enumerate(authors)}
    id_author_map = {idx: author for author, idx in author_id_map.items()}

    #test_data = pd.read_csv(f"./data/dataset/{args.config}-test-topic-ood-llm-dataset.csv")
    test_data = pd.read_csv(f"./data/dataset-{args.resource}-res/test-{args.language_test}-llm-dataset.csv")

    ID_test_data = test_data[test_data['type'] == 'ID']
    ID_test_texts = list(ID_test_data['text'])
    ID_test_labels = np.array([author_id_map[author] for author in list(ID_test_data['author'])])
    assert len(ID_test_texts) == len(ID_test_labels)

    train_data = pd.read_csv(f"./data/dataset-{args.resource}-res/train-{args.language_train}-llm-dataset.csv")
    author_ref_fingerprints_map = {}
    for author in authors:
        author_ref_fingerprints_map[author] = get_ref_fingerprint(args, "entropy", author, train_data,
                                                                max_value)
        
    ID_test_predictions, ID_test_probs = get_entropy_predictions(args, ID_test_data, author_ref_fingerprints_map, max_value)

    pd.DataFrame({
        'ID_test_predictions': ID_test_predictions,
        'ID_test_labels': ID_test_labels,
    }).to_csv(f"./data/results/{get_wandb_run_name(args)}-ID_predictions.csv", index=False)

    no_threshold_ID_results = get_results(ID_test_probs, ID_test_labels, NO_THRESHOLD, author_id_map, id_author_map)
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
            "threshold/ID/all": np.mean(ID_results['f1']),
    }
    wandb.log(threshold_result)
    wandb.run.summary.update(threshold_result)

    wandb.finish()
