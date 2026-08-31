language = "russian"
resource = "high"

import pandas as pd
import pickle
import logging



import numpy as np
from sklearn import preprocessing

logging.basicConfig(level=logging.INFO)


#import jieba
import regex 
import re

def base_preprocessor(string: str) -> str:
    """
    Function that computes regular expressions.
    """
    string = re.sub("[0-9]", "0", string)  # each digit will be represented as a 0
    string = re.sub(r'( \n| \t)+', '', string)
    # text = re.sub("[0-9]+(([.,^])[0-9]+)?", "#", text)
    string = re.sub("https:\\\+([a-zA-Z0-9.]+)?", "@", string)
    return string

def char_diff_preprocessor(string: str) -> str:
    """
    Function that computes regular expressions.
    """
    string = base_preprocessor(string)
    string = re.sub("[a-zA-Z]+", "*", string) ##FOR ENGLISH
    #string = regex.sub(r"\p{L}+", "*", string) ##FOR EVERY OTHER LANGUAGE 
    # string = ''.join(['*' if char.isalpha() else char for char in string])
    return string


def word_preprocessor(string: str) -> str:
    """
    Function that computes regular expressions.
    """
    string = base_preprocessor(string)
    # if model is a word n-gram model, remove all punctuation
    string = ''.join([char for char in string if char.isalnum() or char.isspace()])
    return string


'''#ONLY FOR CHINESE
def word_preprocessor(string: str) -> str:
    string = base_preprocessor(string)
    string = ' '.join(jieba.cut(string))
    string = ''.join([char for char in string if char.isalnum() or char.isspace()])
    return string'''


from sklearn import metrics
def aa_metrics(labels, predictions, raw_outputs, prefix='', no_auc=False, special=False):

    print("Num samples:", len(labels))
    print("Unique labels:", set(labels))
    print("Unique predictions:", set(predictions))

    print("First 20 labels:", labels[:20])
    print("First 20 predictions:", predictions[:20])

    print("All correct?:", all(l == p for l, p in zip(labels, predictions)))

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

    results.update({
        f'{prefix}micro_recall': micro_recall,
        f'{prefix}macro_recall': macro_recall,
        f'{prefix}micro_precision': micro_precision,
        f'{prefix}macro_precision': macro_precision,
        f'{prefix}micro_f1': micro_f1,
        f'{prefix}macro_f1': macro_f1,
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


train_data = pd.read_csv(f"../data/dataset-{resource}-res/train-{language}-llm-dataset.csv")
authors = sorted(list(set(train_data['author'])))
author_id_map = {author: idx for idx, author in enumerate(authors)}


dev_data = pd.read_csv(f'../data/dataset-{resource}-res/devset-{language}.csv')
dev_data.info()
dev_texts = list(dev_data['text'])
dev_labels = np.array([author_id_map[author] for author in list(dev_data['author'])])


def get_ngram_probs(texts, ngram_type):
    if ngram_type == 'char':
        gram_range = [2, 5]
    elif ngram_type == 'dist_char':
        gram_range = [1, 3]
    elif ngram_type == 'word':
        gram_range = [1, 3]
    else:
        raise ValueError(f'ngram_type was not set properly, should be in [char, dist_char, word], got {ngram_type}')
    
    analyzer = ngram_type
    max_features = 100_000
    min_df = 0.01
    sublinear_tf = False
    #name = f"n-gram-{CONFIG}-train-topic-ood-llm-dataset-unseen-None"
    name = f"n-gram-{resource}res-train-{language}-llm-dataset-unseen-None"
    logistic_regression = False

    # cache the vectorizer, just load it if the params match up
    count_vectorizer_cache_path = f'../ngram_cache/cv_{name}_{analyzer}_{gram_range[0]}-{gram_range[1]}_' \
                                  f'{max_features}_{min_df}.pkl'
    tfidf_vectorizer_cache_path = f'../ngram_cache/idf_{name}_{analyzer}_{gram_range[0]}-{gram_range[1]}_' \
                                  f'{max_features}_{min_df}_{sublinear_tf}.pkl'
    with open(count_vectorizer_cache_path, 'rb') as f:
        count_vectorizer = pickle.load(f)
    term_matrix = count_vectorizer.transform(texts)
    with open(tfidf_vectorizer_cache_path, 'rb') as f:
        tfidf_transformer = pickle.load(f)
    data = tfidf_transformer.transform(term_matrix)
    # logging.info(f'{analyzer}: scaling the vectorized data')
    max_abs_scaler = preprocessing.MaxAbsScaler()
    scaled_data = max_abs_scaler.fit_transform(data)
    
    clf_name = 'logreg_sgd' if logistic_regression else 'logreg'
    svm_path = f'../ngram_cache/{analyzer}_{clf_name}-{name}-.pkl'
    with open(svm_path, 'rb') as f:
        classifier = pickle.load(f)
    predicted_probs = classifier.predict_proba(scaled_data)
    return predicted_probs


def get_ngram_predictions(texts):
    probas_word = get_ngram_probs(texts, ngram_type="word")
    probas_dist = get_ngram_probs(texts, ngram_type="dist_char")
    probas_char = get_ngram_probs(texts, ngram_type="char")
    avg_probas = np.average([probas_word, probas_dist, probas_char], axis=0)
    avg_predictions = []
    for text_probs in avg_probas:
        ind_best = np.argmax(text_probs)
        avg_predictions.append(ind_best)
    return avg_predictions, avg_probas


dev_predictions, dev_probas = get_ngram_predictions(dev_texts)

aa_metrics(dev_labels, dev_predictions, dev_probas, prefix="", no_auc=True)


def get_metric_for_threshold(y_true, y_pred_probs):
    n_classes = y_pred_probs.shape[1]
    assert n_classes == len(author_id_map)
    
    # Get top-1 predictions and their probabilities
    top1_probs = np.max(y_pred_probs, axis=1)
    top1_classes = np.argmax(y_pred_probs, axis=1)
    
    thresholds = np.linspace(0.1, 1, 101)
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


result = get_metric_for_threshold(y_true=dev_labels, y_pred_probs=dev_probas)


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
    plt.savefig(f"../data/imgs/{language}-ngram-thresholding.png", dpi=300, bbox_inches='tight')
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
    print("No point where F1 = maxxx")

print(f"First threshold where F1 = {maxxx}:{thresholds[idxs[0]]}")

