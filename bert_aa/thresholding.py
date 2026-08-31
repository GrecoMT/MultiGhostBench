import os
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

LANGUAGE='russian'
RESOURCE='high'
EPOCHS=1

from functools import lru_cache
#from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer

@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')

def split_text_into_chunks(text: str, chunk_size: int = 512):
    tokens = get_tokenizer().encode(text)

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [get_tokenizer().decode(chunk) for chunk in chunks]

    return text_chunks

from sklearn import metrics
def aa_metrics(labels, predictions, raw_outputs, prefix='', no_auc=False, special=False):

    accuracy = metrics.accuracy_score(labels, predictions)
    macro_accuracy = metrics.balanced_accuracy_score(labels, predictions)
    results = {
        f'{prefix}accuracy': accuracy,
        f'{prefix}macro_accuracy': macro_accuracy,
    }
    if special:
        return results

    micro_recall = metrics.recall_score(labels, predictions, average='micro')
    macro_recall = metrics.recall_score(labels, predictions, average='macro')
    micro_precision = metrics.precision_score(labels, predictions, average='micro')
    macro_precision = metrics.precision_score(labels, predictions, average='macro')
    micro_f1 = metrics.f1_score(labels, predictions, average="micro")
    macro_f1 = metrics.f1_score(labels, predictions, average="macro")
    # top2 = metrics.top_k_accuracy_score(labels, raw_outputs, k=2)
    # top3 = metrics.top_k_accuracy_score(labels, raw_outputs, k=3)
    # top4 = metrics.top_k_accuracy_score(labels, raw_outputs, k=4)
    # top5 = metrics.top_k_accuracy_score(labels, raw_outputs, k=5)
    # top6 = metrics.top_k_accuracy_score(labels, raw_outputs, k=6)
    # top7 = metrics.top_k_accuracy_score(labels, raw_outputs, k=7)
    # top8 = metrics.top_k_accuracy_score(labels, raw_outputs, k=8)
    # top9 = metrics.top_k_accuracy_score(labels, raw_outputs, k=9)
    # top10 = metrics.top_k_accuracy_score(labels, raw_outputs, k=10)
    # top25 = metrics.top_k_accuracy_score(labels, raw_outputs, k=25)

    results.update({
        f'{prefix}micro_recall': micro_recall,
        f'{prefix}macro_recall': macro_recall,
        f'{prefix}micro_precision': micro_precision,
        f'{prefix}macro_precision': macro_precision,
        f'{prefix}micro_f1': micro_f1,
        f'{prefix}macro_f1': macro_f1,
        # f'{prefix}top2': top2,
        # f'{prefix}top3': top3,
        # f'{prefix}top4': top4,
        # f'{prefix}top5': top5,
        # f'{prefix}top6': top6,
        # f'{prefix}top7': top7,
        # f'{prefix}top8': top8,
        # f'{prefix}top9': top9,
        # f'{prefix}top10': top10,
        # f'{prefix}top25': top25
    })

    if not no_auc:
        ovr_weighted_auc = metrics.roc_auc_score(labels, raw_outputs, average='weighted', multi_class='ovr')
        ovr_macro_auc = metrics.roc_auc_score(labels, raw_outputs, average='macro', multi_class='ovr')
        ovo_weighted_auc = metrics.roc_auc_score(labels, raw_outputs, average='weighted', multi_class='ovo')
        ovo_macro_auc = metrics.roc_auc_score(labels, raw_outputs, average='macro', multi_class='ovo')



        results.update({
            f'{prefix}ovr_weighted_auc': ovr_weighted_auc,
            f'{prefix}ovr_macro_auc': ovr_macro_auc,
            f'{prefix}ovo_weighted_auc': ovo_weighted_auc,
            f'{prefix}ovo_macro_auc': ovo_macro_auc,
        })

    logging.info(results)
    return results


train_data = pd.read_csv(f"../data/dataset-{RESOURCE}-res/train-{LANGUAGE}-llm-dataset.csv")
authors = sorted(list(set(train_data['author'])))
author_id_map = {author: idx for idx, author in enumerate(authors)}


num_labels = len(author_id_map)
print("num of lavels:", num_labels)

from datasets import tqdm
from transformers import BertForSequenceClassification
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel, ClassificationArgs



# Optional: Move to GPU if available
import torch


def get_bertaa_predictions(texts):
    # Replace with the exact directory path where it was saved (e.g., "./BertAA/your-run-name")

    saved_dir = f"../outputs-bertAA/outputs-{RESOURCE}res-BertAA/{LANGUAGE}/BertAA-train-{LANGUAGE}-llm-dataset-unseen-None-epochs-{EPOCHS}"  # Use the actual folder name created during saving
    model_args = ClassificationArgs(
        process_count=1,          # ← disables multiprocessing, fixes the ForkPoolWorker crash
        use_multiprocessing=False, # ← belt and suspenders
        use_multiprocessing_for_evaluation=False,
        silent=True,
    )
    model = ClassificationModel(
        "xlmroberta",                  # or your model_type, e.g., "roberta"
        saved_dir,   # path to the saved directory
        num_labels=num_labels,   # must match training
        # args={                   # use the SAME args as training (important!)
        #     # Include relevant training_args here, e.g.:
        #     # "overwrite_output_dir": True,
        #     # "no_cuda": params.no_cuda,
        #     # etc.
        # },
        use_cuda=True,
        args=model_args,
        cuda_device=0
    )
    print(model.model)
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


dev_data = pd.read_csv(f'../data/dataset-{RESOURCE}-res/devset-{LANGUAGE}.csv')
dev_texts = list(dev_data['text'])
dev_labels = np.array([author_id_map[author] for author in list(dev_data['author'])])
assert len(dev_texts) == len(dev_labels) == 10

dev_predictions, dev_probas = get_bertaa_predictions(dev_texts)


aa_metrics(dev_labels, dev_predictions, dev_probas, prefix="", no_auc=True)

def get_metric_for_threshold(y_true, y_pred_probs):
    n_classes = y_pred_probs.shape[1]
    assert n_classes == 5
    
    # Get top-1 predictions and their probabilities
    top1_probs = np.max(y_pred_probs, axis=1)
    top1_classes = np.argmax(y_pred_probs, axis=1)
    
    thresholds = np.linspace(0, 1, 101)
    results = []
    
    for threshold in thresholds:
        # Predictions: accept if prob > threshold, else reject as "unseen"
        accepted_mask = top1_probs >= threshold
        
        if np.sum(accepted_mask) == 0:
            primary_accuracy = 0
            primary_precision = 0
            primary_recall = 0
            primary_f1 = 0
        else:
            # Compute per-class metrics, then average
            class_accuracies = []
            class_precisions = []
            class_recalls = []
            class_f1s = []
            
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
            
            # print(class_precision, class_recall, class_f1, class_accuracies)
            # Macro average (unweighted mean across classes)
            macro_accuracy = np.mean(class_accuracies) if class_accuracies else 0
            macro_precision = np.mean(class_precisions) if class_precisions else 0
            macro_recall = np.mean(class_recalls) if class_recalls else 0
            macro_f1 = np.mean(class_f1s) if class_f1s else 0
            
            primary_accuracy = macro_accuracy
            primary_precision = macro_precision
            primary_recall = macro_recall
            primary_f1 = macro_f1
                
        
        coverage = np.mean(accepted_mask)  # Fraction of samples accepted
        
        results.append({
            'threshold': threshold,
            'accuracy': primary_accuracy,
            'precision': primary_precision,
            'recall': primary_recall,
            'f1': primary_f1,
            'coverage': coverage
        })
    
    # # Find optimal threshold based on chosen metric
    # if method == 'f1':
    #     optimal_idx = np.argmax([r['f1'] for r in results])
    # elif method == 'accuracy':
    #     optimal_idx = np.argmax([r['accuracy'] for r in results])
    # elif method == 'precision':
    #     optimal_idx = np.argmax([r['precision'] for r in results])
    # elif method == 'recall':
    #     optimal_idx = np.argmax([r['recall'] for r in results])
    # else:
    #     raise ValueError(f"Unknown method: {method}")
    
    # optimal_idx = np.argmax([r['f1'] for r in results])
    # optimal_threshold = results[optimal_idx]['threshold']
    # optimal_metrics = results[optimal_idx]
    # optimal_metrics['average'] = average
    # print(optimal_threshold, optimal_metrics)
    return  results
    # return optimal_threshold, optimal_metrics, results


result = get_metric_for_threshold(y_true=dev_labels, y_pred_probs=np.array(dev_probas))

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
def plot_threshold_analysis(results):
    """
    Visualize how metrics change with threshold.
    """
    thresholds = [r['threshold'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Performance metrics
    ax1.plot(thresholds, [r['accuracy'] for r in results], label='Accuracy', linewidth=2)
    ax1.plot(thresholds, [r['precision'] for r in results], label='Precision', linewidth=2)
    # ax1.plot(thresholds, [r['recall'] for r in results], label='Recall', linewidth=2)
    ax1.plot(thresholds, [r['f1'] for r in results], label='F1 Score', linewidth=2)
    ax1.set_xlabel('Probability Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Performance Metrics vs Threshold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coverage
    ax2.plot(thresholds, [r['coverage'] for r in results], color='green', linewidth=2)
    ax2.set_xlabel('Probability Threshold', fontsize=12)
    ax2.set_ylabel('Coverage (% Accepted)', fontsize=12)
    ax2.set_title('Prediction Coverage vs Threshold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"../data/imgs/BertAA-{RESOURCE}-{LANGUAGE}-thresholding.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_threshold_analysis(result)



f1_values = np.array([r["f1"] for r in result])
thresholds = np.array([r["threshold"] for r in result])
maxxx = max(f1_values)
print(f"Max f1 value: {maxxx}")
# indices where F1 is exactly 1
idxs = np.where(f1_values == maxxx)[0]

if len(idxs) > 0:
    last_idx = idxs[-1]
    last_threshold = thresholds[last_idx]
    print(f"Last threshold where F1 = {maxxx}:", last_threshold)
else:
    print("No point where F1 = 1")

print(f"First threshold where F1 = {maxxx}:{thresholds[idxs[0]]}")

'''f1_values = np.array([r["f1"] for r in result])
thresholds = np.array([r["threshold"] for r in result])
coverage = np.array([r["coverage"] for r in result])  # <-- devi averla

# prendi solo punti con coverage = 1
mask = coverage == 1.0

f1_valid = f1_values[mask]
thresholds_valid = thresholds[mask]

# massimo F1 tra quelli con coverage = 1
max_f1 = np.max(f1_valid)

# indice (prendo l'ULTIMO per avere threshold più alto possibile)
idx = np.where(f1_valid == max_f1)[0][-1]

best_threshold = thresholds_valid[idx]

print(f"Best threshold (coverage=1): {best_threshold}")
print(f"F1: {max_f1}")'''