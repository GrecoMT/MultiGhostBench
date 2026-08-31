import argparse
from pprint import pformat

import pandas as pd
import pickle
import logging
import wandb
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, accuracy_score

import regex
import re
#import jieba

logging.basicConfig(level=logging.DEBUG)

def get_wandb_run_name(args):
    return f"n-gram-train-{args.resource}res-{args.language}-llm-dataset-threshold-{args.threshold}"


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
    string = re.sub("[a-zA-Z]+", "*", string) # English
    #string = regex.sub(r"\p{L}+", "*", string) ##Other languages
    
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


#ONLY FOR CHINESE
'''def word_preprocessor(string: str) -> str:
    string = base_preprocessor(string)
    string = ' '.join(jieba.cut(string))
    string = ''.join([char for char in string if char.isalnum() or char.isspace()])
    return string'''


def get_ngram_probs(args, texts, ngram_type):
    # print(f"ngram get: {len(texts)}, {ngram_type}")
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
    #name = f"n-gram-{args.config}-train-topic-ood-llm-dataset-unseen-None"
    name = f"n-gram-{args.resource}res-train-{args.language}-llm-dataset-unseen-None"
    
    logistic_regression = False

    # cache the vectorizer, just load it if the params match up
    count_vectorizer_cache_path = f'./ngram_cache/cv_{name}_{analyzer}_{gram_range[0]}-{gram_range[1]}_' \
                                  f'{max_features}_{min_df}.pkl'
    tfidf_vectorizer_cache_path = f'./ngram_cache/idf_{name}_{analyzer}_{gram_range[0]}-{gram_range[1]}_' \
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
    svm_path = f'./ngram_cache/{analyzer}_{clf_name}-{name}-.pkl'
    with open(svm_path, 'rb') as f:
        classifier = pickle.load(f)
    predicted_probs = classifier.predict_proba(scaled_data)
    return predicted_probs


def get_ngram_predictions(args, texts):
    probas_word = get_ngram_probs(args, texts, ngram_type="word")
    probas_dist = get_ngram_probs(args, texts, ngram_type="dist_char")
    probas_char = get_ngram_probs(args, texts, ngram_type="char")
    avg_probas = np.average([probas_word, probas_dist, probas_char], axis=0)
    avg_predictions = []
    for text_probs in avg_probas:
        ind_best = np.argmax(text_probs)
        avg_predictions.append(ind_best)
    return avg_predictions, avg_probas




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
            #'author_type': author_type_map[author],
        })
    # macro_f1 = np.mean(class_f1s) if class_f1s else 0
    return pd.DataFrame(results)


def get_unseen_ngram_probs(args, texts, ngram_type, unseen_author):
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
    #name = f"n-gram-{args.config}-train-topic-ood-llm-dataset-unseen-{unseen_author}"
    name = f"n-gram-{args.resource}res-train-{args.language}-llm-dataset-unseen-{unseen_author}"
    logistic_regression = False

    # cache the vectorizer, just load it if the params match up
    count_vectorizer_cache_path = f'./ngram_cache/cv_{name}_{analyzer}_{gram_range[0]}-{gram_range[1]}_' \
                                  f'{max_features}_{min_df}.pkl'
    tfidf_vectorizer_cache_path = f'./ngram_cache/idf_{name}_{analyzer}_{gram_range[0]}-{gram_range[1]}_' \
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
    svm_path = f'./ngram_cache/{analyzer}_{clf_name}-{name}-.pkl'
    with open(svm_path, 'rb') as f:
        classifier = pickle.load(f)
    predicted_probs = classifier.predict_proba(scaled_data)
    return predicted_probs


def get_unseen_ngram_predictions(args, texts, unseen_author):
    probas_word = get_unseen_ngram_probs(args, texts, ngram_type="word", unseen_author=unseen_author)
    probas_dist = get_unseen_ngram_probs(args, texts, ngram_type="dist_char", unseen_author=unseen_author)
    probas_char = get_unseen_ngram_probs(args, texts, ngram_type="char", unseen_author=unseen_author)
    avg_probas = np.average([probas_word, probas_dist, probas_char], axis=0)
    avg_predictions = []
    for text_probs in avg_probas:
        ind_best = np.argmax(text_probs)
        avg_predictions.append(ind_best)
    return avg_predictions, avg_probas


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
        #'author_type': author_type_map[unseen_author]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a N-Gram model from the command line')
    #parser.add_argument('--config', type=str)
    parser.add_argument('--language', type=str)

    parser.add_argument('--resource', type=str)

    parser.add_argument('--threshold', type=str)
    args = parser.parse_args()
    args.threshold = float(args.threshold)
    logging.info("Args:\n%s", pformat(args))
    wandb.login()
    wandb.init(project="multilingual-OOD-AA", name=get_wandb_run_name(args), config=args)
    THRESHOLD = args.threshold
    '''dataset = pd.read_csv(
        f"./data/dataset/{args.config}-train-topic-ood-llm-dataset.csv")'''
    dataset = pd.read_csv(
        f"./data/dataset-{args.resource}-res/train-{args.language}-llm-dataset.csv")

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

    ID_test_predictions, ID_test_probs = get_ngram_predictions(args, ID_test_texts)
    OOD_test_predictions, OOD_test_probs = get_ngram_predictions(args, OOD_test_texts)

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
        unseen_author_labels, unseen_author_probs = get_unseen_ngram_predictions(args, unseen_author_texts, unseen_author)
        unseen_results.append(get_unseen_results(unseen_author_probs, THRESHOLD, unseen_author))
    unseen_results = pd.DataFrame(unseen_results)

    print("Unseen Author:")
  
    print("all: ", np.mean(unseen_results['f1']))
   
    threshold_result = {
            "threshold/ID/all": np.mean(ID_results['f1']),
            "threshold/OOD/all": np.mean(OOD_results['f1']),
            "threshold/Unseen/all": np.mean(unseen_results['f1']),

    }
    unseen_results.to_csv(f"./data/results/{get_wandb_run_name(args)}-threshold-{THRESHOLD}_unseen_results.csv", index=False)
    wandb.log(threshold_result)
    wandb.run.summary.update(threshold_result)

    wandb.finish()
