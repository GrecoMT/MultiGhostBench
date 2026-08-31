RESOURCE = "low"
LANGUAGE = "russian"

import os
import matplotlib
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
import os
import pickle
import numpy as np
from src.index import Indexer
import torch
import argparse
from src.text_embedding import TextEmbeddingModel
import random
import numpy as np
from src.utils import OOD_utils


logging.basicConfig(level=logging.ERROR)


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


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


#model_name = "princeton-nlp/unsup-simcse-roberta-base"
model_name = "ZurichNLP/unsup-simcse-xlm-roberta-base"
#model_path = f"../DeTeCtive/src/runs/{CONFIG}-LLM-OOD-AA-detective-unseen-None_v0/model_best.pth"
model_path = f"./src/runs/{RESOURCE}-{LANGUAGE}-OOD-unseen-None_v0/model_best.pth"
embedding_dim = 768
database_path = f"./src/database/database-{RESOURCE}-{LANGUAGE}-unseen-None" 
K = 5

model = TextEmbeddingModel(model_name).cuda()
state_dict = torch.load(model_path, map_location=model.model.device)
new_state_dict={}
for key in state_dict.keys():
    if key.startswith('model.'):
        new_state_dict[key[6:]]=state_dict[key]
model.load_state_dict(state_dict)
tokenizer=model.tokenizer

index = Indexer(embedding_dim)
index.deserialize_from(database_path)
label_dict=load_pkl(os.path.join(database_path,'label_dict.pkl'))


from collections import Counter
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

def process_top_ids_and_scores_AA(top_ids_and_scores, label_dict):
    preds=[]
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        num_dict={}
        max_num,max_id=0,0
        for id in ids:
            if label_dict[int(id)] not in num_dict:
                num_dict[label_dict[int(id)]]=1
            else:
                num_dict[label_dict[int(id)]]+=1
            if num_dict[label_dict[int(id)]]>max_num:
                max_num=num_dict[label_dict[int(id)]]
                max_id=label_dict[int(id)]
        preds.append(str(max_id))
    return preds

def get_detective_prediction_per_text(chunks, batch_size=128, show_progress=False):
    model.eval()
    
    all_embeddings = []
    
    # Create DataLoader for automatic batching & memory management
    loader = DataLoader(
        chunks,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,           # tune or set 0 on windows if issues
        collate_fn=lambda x: x   # identity -> we encode inside loop
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

def get_detective_predictions(texts):
    predictions = []
    raw_outputs = []
    for text in tqdm(texts, desc="Predicting for long texts"):
        chunks = OOD_utils.split_text_into_chunks(text)
        # print("chunks: ", len(chunks))
        majority_class, percentages = get_detective_prediction_per_text(chunks)
        assert majority_class == np.argmax(percentages), f"{majority_class}, {percentages}"
        predictions.append(majority_class)
        raw_outputs.append(percentages)
    return predictions, raw_outputs



#train_data = pd.read_csv(f"../data/dataset/{CONFIG}-train-topic-ood-llm-dataset.csv")
train_data = pd.read_csv(f"../data/dataset-{RESOURCE}-res/train-{LANGUAGE}-llm-dataset.csv")
authors = sorted(list(set(train_data['author'])))
author_id_map = {author: idx for idx, author in enumerate(authors)}
#author_id_map

num_labels = len(author_id_map)
print("num of lavels:", num_labels)

#dev_data = pd.read_csv('../data/dataset/dev-time-ood-llm-dataset.csv')
dev_data = pd.read_csv(f'../data/dataset-{RESOURCE}-res/devset-{LANGUAGE}.csv')
dev_texts = list(dev_data['text'])
dev_labels = np.array([author_id_map[author] for author in list(dev_data['author'])])
#assert len(dev_texts) == len(dev_labels) == 20
assert len(dev_texts) == len(dev_labels) == 10

train_texts = list(train_data['text'])
train_labels = np.array([author_id_map[author] for author in list(train_data['author'])])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    avg_f1 = f1_score(y_true, y_pred, average='macro')
    avg_recall = recall_score(y_true, y_pred, average='macro')
    return accuracy, avg_f1,avg_recall


dev_predictions, dev_probas = get_detective_predictions(dev_texts)
accuracy, avg_f1,avg_rec=calculate_metrics(dev_labels, dev_predictions)
print(f"Dev Accuracy: {accuracy}, AvgF1: {avg_f1}, AvgRecall: {avg_rec}")


aa_metrics(dev_labels, dev_predictions, dev_probas, prefix="", no_auc=True)


def get_metric_for_threshold(y_true, y_pred_probs):
    n_classes = y_pred_probs.shape[1]
    #assert n_classes == 10
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
    plt.savefig(f"../data/imgs/Detective-{RESOURCE}-res-{LANGUAGE}-thresholding.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_threshold_analysis(result)


f1_values = np.array([r["f1"] for r in result])
thresholds = np.array([r["threshold"] for r in result])
coverage_values = np.array([r["coverage"] for r in result])

MIN_COVERAGE = 0.80

valid_indices = np.where(coverage_values >= MIN_COVERAGE)[0]

if len(valid_indices) > 0:
    valid_f1_values = f1_values[valid_indices]
    max_valid_f1 = np.max(valid_f1_values)

    best_indices = valid_indices[
        np.where(np.isclose(valid_f1_values, max_valid_f1))[0]
    ]

    # Prende la threshold più bassa sul miglior plateau:
    # stessa F1, ma scelta più conservativa e stabile.
    selected_idx = best_indices[0]

    print(f"Selected threshold: {thresholds[selected_idx]}")
    print(f"F1 at selected threshold: {f1_values[selected_idx]}")
    print(f"Coverage at selected threshold: {coverage_values[selected_idx]}")
else:
    print(f"No threshold satisfies minimum coverage {MIN_COVERAGE}")