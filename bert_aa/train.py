import os
import gc
import argparse
import logging
import resource
import torch
import pandas as pd
from datasets import tqdm
from sklearn.metrics import f1_score, accuracy_score
from simpletransformers.classification import ClassificationModel
import numpy as np
from utils.datasets import load_dataset_valla_chunking
from functools import lru_cache
from transformers import AutoTokenizer, BertForSequenceClassification
from scipy.special import softmax



rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)



from utils import metrics, datasets

def get_run_name(args):
    return (
        f"BertAA-"
        f"{args.train_dataset_name}-"
        f"{args.resource}-res-"
        f"epochs-{args.num_train_epochs}"
    )

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

@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')

def split_text_into_chunks(text: str, chunk_size: int = 512):
    tokens = get_tokenizer().encode(text)

    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [get_tokenizer().decode(chunk) for chunk in chunks]

    return text_chunks

def get_bertaa_predictions(model, texts):
    predictions = []
    raw_outputs = []
    for text in tqdm(texts, total=len(texts), desc="Predicting"):
        chunks = split_text_into_chunks(text)
        chunk_preds, chunk_raw = model.predict(chunks)
        chunk_probs = [softmax(logits) for logits in chunk_raw]
        avg_probs = np.mean(chunk_probs, axis=0)
        predictions.append(np.argmax(avg_probs))
        raw_outputs.append(avg_probs)
    return predictions, np.array(raw_outputs)

def get_model(model_path, num_labels, training_args, use_cuda=False, cuda_device=0, only_train_classifier=False):
    if only_train_classifier:
        # set the custom_parameter_groups arg properly, as well as the
        # train_custom_parameters_only=True
        logging.info(f'training only the classification layer')
        training_args.custom_parameter_groups = [{
            'params': ['classifier.weight', 'classifier.bias'],
            'lr': training_args["learning_rate"]
        }]
        training_args.train_custom_parameters_only = True
    training_args["no_cache"] = True
    logging.info(f'setting lr: {training_args["learning_rate"]}')

    model = ClassificationModel('bert' if 'roberta' not in model_path else 'xlmroberta',
                                model_path,
                                num_labels=num_labels,
                                args=training_args,
                                use_cuda=True,
                                cuda_device=cuda_device
                                )
    return model


def get_training_args(params):

    training_args = {
        'reprocess_input_data': params.reprocess_input_data,  # reprocess the input data
        'num_train_epochs': params.num_train_epochs,  # number of epochs
        'evaluate_during_training': params.evaluate_during_training,  # run evaluation during training
        "use_early_stopping": params.use_early_stopping,
        "early_stopping_consider_epochs": params.early_stopping_consider_epochs,
        'evaluate_during_training_steps': params.evaluate_during_training_steps,  # steps in training before eval
        'evaluate_each_epoch': params.evaluate_each_epoch,
        'train_batch_size': params.train_batch_size,  # training batch size
        "eval_batch_size": params.eval_batch_size,  # evaluation batch size
        "gradient_accumulation_steps": params.gradient_accumulation_steps,  # steps before applying gradients
        "save_eval_checkpoints": params.save_eval_checkpoints,  # save evaluation checkpoints
        "learning_rate": params.lr,  # learning rate of our model
        "max_seq_length": params.max_seq_len,  # maximum sequence length in tokens
        "sliding_window": params.sliding_window,
        "stride": int(params.doc_stride * params.max_seq_len),  # stride when processing sentences
        "logging_steps": 1,  # the number of steps before logging
        "warmup_ratio": params.warmup_ratio,
        "warmup_steps": params.warmup_steps,
        "weight_decay": params.weight_decay,
        "manual_seed": params.seed,  # set the random seed
        "lazy_loading": params.lazy_loading,
        "lazy_labels_column": 1,
        "lazy_text_column": 0,
        "lazy_loading_start_line": 0,
        "dataloader_num_workers": params.dataloader_num_workers,
        "overwrite_output_dir": params.overwrite_output_dir,
        'output_dir': f"./outputs/{get_run_name(params)}/",
        "save_steps": -1,
        "save_model_every_epoch": False,
        "save_eval_checkpoints": False
    }
    return training_args


def tuning(args):
    
    #TRAIN
    df_train = datasets.load_dataset_valla_chunking(args.train_dataset_name, args.resource, args.unseen_author)[0]
    df_train = datasets.aa_as_pandas(df_train)

    train_labels = df_train['labels'].unique()
    num_labels = df_train['labels'].nunique()
    
    #AUTHOR ID MAP AND REVERSE, not important which dataset is used
    dataset = pd.read_csv(f"./data/dataset-{args.resource}-res/{args.dev_dataset_name}.csv")
    authors = sorted(list(set(dataset['author'])))
    logging.info(f'Number of authors: {len(authors)}')
    logging.info(f'Authors: {authors}')
    author_id_map = {author: idx for idx, author in enumerate(authors)}
    id_author_map = {idx: author for author, idx in author_id_map.items()}

    #DEV
    dev_data = pd.read_csv(f"./data/dataset-{args.resource}-res/{args.dev_dataset_name}.csv")

    dev_texts = list(dev_data['text'])
    dev_labels = np.array([author_id_map[author] for author in list(dev_data['author'])])

    #results
    epoch_results = []

    for epochs in args.epoch_values:
        logging.info(f"TRAINING WITH EPOCHS = {epochs}")
        args.num_train_epochs = epochs
        print(f"NUM_TRAIN_EPOCHS {args.num_train_epochs}")
        
        training_args = get_training_args(args)
        
        model = get_model(model_path=args.model_path, num_labels=num_labels, training_args=training_args,
                      use_cuda=not args.no_cuda, cuda_device=args.device)
    
        print(model.model)

        #train
        global_steps, train_loss = model.train_model(df_train)

        #predict on dev
        dev_predictions, dev_probs = get_bertaa_predictions(model, dev_texts)


        per_class_results = get_results(
            dev_probs,
            dev_labels,
            threshold=0,
            author_id_map=author_id_map,
            id_author_map=id_author_map
        )

        dev_macro_f1 = per_class_results['f1'].mean()
        dev_accuracy = accuracy_score(dev_labels, dev_predictions)

        result = {
            "epochs": epochs,
            "train_loss": train_loss,
            "dev_macro_f1": dev_macro_f1,
            "dev_accuracy": dev_accuracy,
            "output_dir": training_args["output_dir"],
        }

        epoch_results.append(result)

        logging.info(
            f"epochs={epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"macro_f1={dev_macro_f1:.4f} | "
            f"accuracy={dev_accuracy:.4f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    results_df = pd.DataFrame(epoch_results)

    best_row = results_df.loc[results_df["dev_macro_f1"].idxmax()]
    best_epochs = int(best_row["epochs"])

    print("\n========== EPOCH TUNING RESULTS ==========")
    print(results_df[["epochs", "train_loss", "dev_macro_f1", "dev_accuracy"]].to_string(index=False))

    print("\n========== BEST CONFIG ==========")
    print(f"Best epochs: {best_epochs}")
    print(f"Best train loss: {best_row['train_loss']:.4f}")
    print(f"Best dev macro-F1: {best_row['dev_macro_f1']:.4f}")
    print(f"Best dev accuracy: {best_row['dev_accuracy']:.4f}")
    print(f"Best model dir: {best_row['output_dir']}")

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(
        args.results_dir,
        f"epoch_tuning_{args.train_dataset_name}_{args.resource}_res.csv"
    )
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")

    return best_epochs, results_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning on epochs number for XLM-RoBeRTa-AA"
    )

    parser.add_argument("--train_dataset_name", type=str, required=True)
    parser.add_argument("--dev_dataset_name", type=str, required=True)

    parser.add_argument("--resource", type=str, required=True)
    parser.add_argument("--unseen_author", type=str, default=None)

    parser.add_argument(
        "--epoch_values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 7, 10]
    )

    parser.add_argument("--model_path", type=str, default="FacebookAI/xlm-roberta-large")
    #parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--evaluate_during_training', action='store_true')
    parser.add_argument('--evaluate_during_training_steps', type=int, default=-1)
    parser.add_argument('--evaluate_each_epoch', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.15)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--doc_stride', type=float, default=0.8, help='express as % of max_seq_len')
    parser.add_argument('--use_early_stopping', action='store_true')
    parser.add_argument('--early_stopping_consider_epochs', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--lazy_loading', action='store_true')
    parser.add_argument('--only_train_classifier', action='store_true')
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    parser.add_argument("--results_dir", type=str, default="./data/XLM-AA-tuning")

    parser.set_defaults(final_run=False, early_stopping_metric_minimize=False, early_stopping_consider_epochs=False,
                        use_early_stopping=False, overwrite_output_dir=True, save_best_model=False,
                        save_model_every_epoch=False, save_eval_checkpoints=False, reprocess_input_data=False,
                        evaluate_during_training=False, sliding_window=False, no_cuda=False,
                        only_train_classifier=False, lazy_loading=False)


    args = parser.parse_args()

    tuning(args)