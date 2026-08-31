import torch
import transformers

RESOURCE = "high"
LANGUAGE = "russian"
METHOD = "Rank"
UNSEEN_AUTHOR = None
DEVICE = "cuda"
CACHE_DIR = "cache"
EVAL_MODEL = "ai-forever/mGPT" #TODO 

import pandas as pd
import pickle
import logging

import numpy as np
from sklearn import preprocessing
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)

def get_entropy(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_rank(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
            1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        # assert (timesteps == torch.arange(len(timesteps)).to(
        #     timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


def get_rank_GLTR(text, base_model, base_tokenizer, DEVICE, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
            1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        # assert (timesteps == torch.arange(len(timesteps)).to(
        #     timesteps.device)).all(), "Expected one match per timestep"
        ranks = ranks.float()
        res = np.array([0.0, 0.0, 0.0, 0.0])
        for i in range(len(ranks)):
            if ranks[i] < 10:
                res[0] += 1
            elif ranks[i] < 100:
                res[1] += 1
            elif ranks[i] < 1000:
                res[2] += 1
            else:
                res[3] += 1
        if res.sum() > 0:
            res = res / res.sum()
        # print(res)
        return res


def entropy_criterion(text): return get_entropy(
    text, base_model, base_tokenizer, DEVICE)

def GLTR_criterion(text): return get_rank_GLTR(
    text, base_model, base_tokenizer, DEVICE)

def rank_criterion(text): return -get_rank(text,
                                       base_model, base_tokenizer, DEVICE, log=False)


def get_wandb_run_name():
    #return f"{METHOD}-{CONFIG}-{EVAL_MODEL}-OOD-unseen-{UNSEEN_AUTHOR}"
    model = EVAL_MODEL.replace("/", "-")
    return f"{METHOD}-{RESOURCE}res-{LANGUAGE}-{model}-ID-unseen-{UNSEEN_AUTHOR}"


def load_base_model_and_tokenizer(name, cache_dir):

    print(f'Loading BASE model {name}...')
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        name, cache_dir=cache_dir)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer

def load_base_model(base_model, DEVICE):

    base_model.to(DEVICE)


base_model, base_tokenizer = load_base_model_and_tokenizer(
EVAL_MODEL, CACHE_DIR)
load_base_model(base_model, DEVICE)



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


train_data = pd.read_csv(f"../data/dataset-{RESOURCE}-res/train-{LANGUAGE}-llm-dataset.csv")
authors = sorted(list(set(train_data['author'])))
author_id_map = {author: idx for idx, author in enumerate(authors)}

author_id_map.keys()

dev_data = pd.read_csv(f'../data/dataset-{RESOURCE}-res/devset-{LANGUAGE}.csv')
dev_texts = list(dev_data['text'])
dev_labels = np.array([author_id_map[author] for author in list(dev_data['author'])])


from functools import lru_cache
#from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer
from tqdm import tqdm

@lru_cache(maxsize=1)
def get_tokenizer():
    #return GPT2TokenizerFast.from_pretrained("gpt2")
    return AutoTokenizer.from_pretrained("ai-forever/mGPT")

def split_text_into_chunks(text: str, chunk_size: int = 512):
    tokens = get_tokenizer().encode(text)

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [get_tokenizer().decode(chunk) for chunk in chunks]

    return text_chunks

def get_mgt_predictions(texts):
    with open(f'../clf_results/{get_wandb_run_name()}_logistic_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    y_test_pred_prob = []
    y_test_pred = []
    for idx in tqdm(range(len(texts))):
        curr_text = texts[idx]
        chunks = split_text_into_chunks(curr_text)
        if METHOD == 'Rank':
            chunks_criterion = np.array([rank_criterion(chunk) for chunk in chunks])
        elif METHOD == 'Entropy':
            chunks_criterion = np.array([entropy_criterion(chunk) for chunk in chunks])
        elif METHOD == 'GLTR':
            chunks_criterion = np.array([GLTR_criterion(chunk) for chunk in chunks])
        else:
            raise Exception("Not supported yet")
        if METHOD != "GLTR":
            select_index = ~np.isnan(chunks_criterion)
            chunks_criterion = chunks_criterion[select_index]
        print(chunks_criterion.shape)
        if METHOD != "GLTR":
            chunks_criterion = np.expand_dims(chunks_criterion, axis=-1)
        chunk_probs = clf.predict_proba(chunks_criterion)  # Probability for positive class

        final_prob = np.mean(chunk_probs, axis=0)
        y_test_pred_prob.append(final_prob)
        y_test_pred.append(np.argmax(final_prob))
    return y_test_pred, y_test_pred_prob

train_texts = list(train_data['text'])
train_labels = np.array([author_id_map[author] for author in list(train_data['author'])])


dev_predictions, dev_probas = get_mgt_predictions(dev_texts)


aa_metrics(dev_labels, dev_predictions, dev_probas, prefix="", no_auc=True)



def get_metric_for_threshold(y_true, y_pred_probs):
    y_pred_probs = np.array(y_pred_probs)
    n_classes = y_pred_probs.shape[1]
    assert n_classes == len(author_id_map)
    
    # Get top-1 predictions and their probabilities
    top1_probs = np.max(y_pred_probs, axis=1)
    top1_classes = np.argmax(y_pred_probs, axis=1)
    
    thresholds = np.linspace(0.1, 0.5, 101)
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
    #plt.savefig(f"../data/imgs/{CONFIG}-{METHOD}-thresholding.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"../data/imgs/{RESOURCE}res-{LANGUAGE}-{METHOD}-thresholding.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_threshold_analysis(result)

f1_values = np.array([r["f1"] for r in result])
thresholds = np.array([r["threshold"] for r in result])
coverage_values = np.array([r["coverage"] for r in result])

max_f1 = np.max(f1_values)

# indici dove F1 è massimo
idxs = np.where(np.isclose(f1_values, max_f1))[0]

# primo intervallo consecutivo
start_idx = idxs[0]
end_idx = start_idx

for idx in idxs[1:]:
    if idx == end_idx + 1:
        end_idx = idx
    else:
        break

print(f"Max F1: {max_f1:.4f}")
print(f"First max-F1 threshold interval: [{thresholds[start_idx]:.4f}, {thresholds[end_idx]:.4f}]")
print(f"Coverage interval: [{coverage_values[start_idx]:.4f}, {coverage_values[end_idx]:.4f}]")