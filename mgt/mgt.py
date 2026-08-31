import argparse
from pprint import pformat

import pandas as pd
import pickle
import logging

import torch
import transformers
from tqdm import tqdm

import wandb
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, accuracy_score
from functools import lru_cache
#from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F
from utils.metrics import aa_metrics

import re

logging.basicConfig(level=logging.ERROR)
DEVICE = "cuda"
CACHE_DIR = "cache"
#EVAL_MODEL = "gpt2-medium"
EVAL_MODEL = "ai-forever/mGPT"

def sanitize_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', "-", str(name))

def get_run_name(args, unseen_author):
    EVAL_MODEL_SANITIZED = sanitize_filename(EVAL_MODEL)
    return f"{args.method}-{args.resource}res-{args.language}-{EVAL_MODEL_SANITIZED}-ID-unseen-{unseen_author}"  # OOD hard coded; doesnt matter

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


def get_wandb_run_name(args):
    return f"mgt-{args.method}-{args.resource}res-{args.language}-train-llm-dataset-threshold-{args.threshold}"

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

@lru_cache(maxsize=1)
def get_tokenizer():
    #return GPT2TokenizerFast.from_pretrained("gpt2")
    return AutoTokenizer.from_pretrained("ai-forever/mGPT")

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

        return res

def entropy_criterion(text): return get_entropy(
    text, base_model, base_tokenizer, DEVICE)

def GLTR_criterion(text): return get_rank_GLTR(
    text, base_model, base_tokenizer, DEVICE)

def rank_criterion(text): return -get_rank(text,
                                       base_model, base_tokenizer, DEVICE, log=False)



def split_text_into_chunks(text: str, chunk_size: int = 512):
    tokens = get_tokenizer().encode(text)

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [get_tokenizer().decode(chunk) for chunk in chunks]

    return text_chunks

def get_mgt_predictions(args, texts, unseen_author=None):
    print("unseen author", unseen_author)
    with open(f'./clf_results/{get_run_name(args, unseen_author)}_logistic_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    y_test_pred_prob = []
    y_test_pred = []
    for idx in tqdm(range(len(texts))):
        curr_text = texts[idx]
        chunks = split_text_into_chunks(curr_text)
        if args.method == 'Rank':
            chunks_criterion = np.array([rank_criterion(chunk) for chunk in chunks])
        elif args.method == 'Entropy':
            chunks_criterion = np.array([entropy_criterion(chunk) for chunk in chunks])
        elif args.method == 'GLTR':
            chunks_criterion = np.array([GLTR_criterion(chunk) for chunk in chunks])
        else:
            raise Exception("Not supported yet")
        if args.method != "GLTR":
            select_index = ~np.isnan(chunks_criterion)
            chunks_criterion = chunks_criterion[select_index]
        if args.method != "GLTR":
            chunks_criterion = np.expand_dims(chunks_criterion, axis=-1)
        chunk_probs = clf.predict_proba(chunks_criterion)  # Probability for positive class

        final_prob = np.mean(chunk_probs, axis=0)
        y_test_pred_prob.append(final_prob)
        y_test_pred.append(np.argmax(final_prob))
    return y_test_pred, y_test_pred_prob


'''def get_results(y_pred_probs, y_true, threshold, author_id_map, id_author_map, author_type_map):
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
            'author_type': author_type_map[author],
        })
    # macro_f1 = np.mean(class_f1s) if class_f1s else 0
    return pd.DataFrame(results)'''

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
        'samples': len(top1_probs)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a N-Gram model from the command line')


    #parser.add_argument('--config', type=str)
    parser.add_argument('--language', type=str)
    parser.add_argument('--resource', type=str)
    
    
    parser.add_argument('--method', type=str)
    parser.add_argument('--threshold', type=str)
    args = parser.parse_args()
    args.threshold = float(args.threshold)
    logging.info("Args:\n%s", pformat(args))
    wandb.login()
    wandb.init(project="LLM-OOD-AA", name=get_wandb_run_name(args), config=args)
    THRESHOLD = args.threshold
    #dataset = pd.read_csv(f"./data/dataset/{args.config}-train-topic-ood-llm-dataset.csv")
    dataset = pd.read_csv(f"./data/dataset-{args.resource}-res/train-{args.language}-llm-dataset.csv")

    authors = sorted(list(set(dataset['author'])))
    logging.info(f'Number of authors: {len(authors)}')
    logging.info(f'Authors: {authors}')
    author_id_map = {author: idx for idx, author in enumerate(authors)}
    id_author_map = {idx: author for author, idx in author_id_map.items()}

    test_data = pd.read_csv(f"./data/dataset-{args.resource}-res/test-{args.language}-llm-dataset.csv")

    ID_test_data = test_data[test_data['type'] == 'ID']
    ID_test_texts = list(ID_test_data['text'])
    ID_test_labels = np.array([author_id_map[author] for author in list(ID_test_data['author'])])
    assert len(ID_test_texts) == len(ID_test_labels)

    OOD_test_data = test_data[test_data['type'] == 'OOD']
    OOD_test_texts = list(OOD_test_data['text'])
    OOD_test_labels = np.array([author_id_map[author] for author in list(OOD_test_data['author'])])
    assert len(OOD_test_texts) == len(OOD_test_labels)

    ID_test_predictions, ID_test_probs = get_mgt_predictions(args, ID_test_texts)
    OOD_test_predictions, OOD_test_probs = get_mgt_predictions(args, OOD_test_texts)

    pd.DataFrame({
        'ID_test_predictions': ID_test_predictions,
        'ID_test_labels': ID_test_labels,
    }).to_csv(f"./data/results/{get_wandb_run_name(args)}-ID_predictions.csv", index=False)
    
    OOD_predictions = pd.DataFrame({
        'OOD_test_predictions': OOD_test_predictions,
        'OOD_test_labels': OOD_test_labels,
    }).to_csv(f"./data/results/{get_wandb_run_name(args)}-OOD_predictions.csv", index=False)

    no_threshold_ID_results = get_results(ID_test_probs, ID_test_labels, 0, author_id_map, id_author_map)
    print("\nNO THRESHOLD:")
    print("ID:")
    print("all: ", np.mean(no_threshold_ID_results['f1']))

    no_threshold_OOD_results = get_results(OOD_test_probs, OOD_test_labels, 0, author_id_map, id_author_map)
    print("OOD:")
    print("all: ", np.mean(no_threshold_OOD_results['f1']))

    no_threshold_result = {
            "no_threshold/ID/all": np.mean(no_threshold_ID_results['f1']),
            "no_threshold/OOD/all": np.mean(no_threshold_OOD_results['f1'])
    }
    no_threshold_ID_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-no_threshold_ID_results.csv", index=False)
    no_threshold_OOD_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-no_threshold_OOD_results.csv",
                                   index=False)
    wandb.log(no_threshold_result)
    wandb.run.summary.update(no_threshold_result)

    print("\nTHRESHOLD:", THRESHOLD)
    ID_results = get_results(ID_test_probs, ID_test_labels, THRESHOLD, author_id_map, id_author_map)
    print("ID:")

    print("all: ", np.mean(ID_results['f1']))
    OOD_results = get_results(OOD_test_probs, OOD_test_labels, THRESHOLD, author_id_map, id_author_map)
    print("OOD:")
    print("all: ", np.mean(OOD_results['f1']))


    ID_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_ID_results.csv",
                                   index=False)
    OOD_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_OOD_results.csv",
                                    index=False)
    unseen_results = []
    for unseen_author in authors:
        print("Unseen author: ", unseen_author)
        curr_authors = authors.copy()
        curr_authors.remove(unseen_author)
        assert len(curr_authors) == 4
        curr_author_id_map = {author: idx for idx, author in enumerate(curr_authors)}
        unseen_author_texts = list(test_data[test_data['author'] == unseen_author]['text'])
        unseen_author_labels, unseen_author_probs = get_mgt_predictions(args, unseen_author_texts, unseen_author)
        
        unseen_results.append(get_unseen_results(unseen_author_probs, THRESHOLD, unseen_author))
    unseen_results = pd.DataFrame(unseen_results)

    print("Unseen:")

    print("all: ", np.mean(unseen_results['f1']))

    threshold_result = {
            "threshold/ID/all": np.mean(ID_results['f1']),
            "threshold/OOD/all": np.mean(OOD_results['f1']),
            "threshold/Unseen/all": np.mean(unseen_results['f1']),
    }
    unseen_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_unseen_results.csv",
                       index=False)
    wandb.log(threshold_result)
    wandb.run.summary.update(threshold_result)

    wandb.finish()
