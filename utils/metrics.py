import logging

import transformers
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import time
from functools import wraps
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# from: https://github.com/JacobTyo/Valla/blob/main/valla/utils/eval_metrics.py
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
    top2 = metrics.top_k_accuracy_score(labels, raw_outputs, k=2)
    top3 = metrics.top_k_accuracy_score(labels, raw_outputs, k=3)
    top4 = metrics.top_k_accuracy_score(labels, raw_outputs, k=4)
    top5 = metrics.top_k_accuracy_score(labels, raw_outputs, k=5)
    top6 = metrics.top_k_accuracy_score(labels, raw_outputs, k=6)
    top7 = metrics.top_k_accuracy_score(labels, raw_outputs, k=7)
    top8 = metrics.top_k_accuracy_score(labels, raw_outputs, k=8)
    top9 = metrics.top_k_accuracy_score(labels, raw_outputs, k=9)
    top10 = metrics.top_k_accuracy_score(labels, raw_outputs, k=10)
    top25 = metrics.top_k_accuracy_score(labels, raw_outputs, k=25)

    results.update({
        f'{prefix}micro_recall': micro_recall,
        f'{prefix}macro_recall': macro_recall,
        f'{prefix}micro_precision': micro_precision,
        f'{prefix}macro_precision': macro_precision,
        f'{prefix}micro_f1': micro_f1,
        f'{prefix}macro_f1': macro_f1,
        f'{prefix}top2': top2,
        f'{prefix}top3': top3,
        f'{prefix}top4': top4,
        f'{prefix}top5': top5,
        f'{prefix}top6': top6,
        f'{prefix}top7': top7,
        f'{prefix}top8': top8,
        f'{prefix}top9': top9,
        f'{prefix}top10': top10,
        f'{prefix}top25': top25
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

# from: https://github.com/xinleihe/MGTBench/blob/main/methods/utils.py
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds\n\n')
        return result
    return timeit_wrapper



def get_clf_results(x_train, y_train, x_test, y_test):

    clf = LogisticRegression(random_state=0).fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_train_pred_prob = clf.predict_proba(x_train)
    y_train_pred_prob = [_[1] for _ in y_train_pred_prob]
    acc_train, precision_train, recall_train, f1_train, auc_train = cal_metrics(
        y_train, y_train_pred, y_train_pred_prob)
    train_res = acc_train, precision_train, recall_train, f1_train, auc_train

    y_test_pred = clf.predict(x_test)
    y_test_pred_prob = clf.predict_proba(x_test)
    y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
    acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(
        y_test, y_test_pred, y_test_pred_prob)
    test_res = acc_test, precision_test, recall_test, f1_test, auc_test

    return clf, train_res, test_res

def cal_metrics(label, pred_label, pred_posteriors):
    if len(set(label)) < 3:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = -1.0
        # auc = roc_auc_score(label, pred_posteriors)
    else:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average='macro')
        recall = recall_score(label, pred_label, average='macro')
        f1 = f1_score(label, pred_label, average='macro')
        auc = -1.0
        conf_m = confusion_matrix(label, pred_label)
        print(conf_m)
    return acc, precision, recall, f1, auc


def load_base_model_and_tokenizer(name, cache_dir):

    print(f'Loading BASE model {name}...')
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        name, cache_dir=cache_dir)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer

def load_base_model(base_model, DEVICE):
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()

    base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')