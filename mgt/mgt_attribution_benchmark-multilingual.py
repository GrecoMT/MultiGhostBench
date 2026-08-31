import argparse
import datetime
import logging
import os

import torch
import re

import wandb
import json
# from methods.utils import load_base_model, load_base_model_and_tokenizer, filter_test_data
# from methods.supervised import run_supervised_experiment
# from methods.detectgpt import run_perturbation_experiments
# from methods.gptzero import run_gptzero_experiment
from utils.metrics import load_base_model_and_tokenizer, load_base_model
from utils.datasets import load_dataset_mgt
from mgt_metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, run_GLTR_experiment

logging.basicConfig(level=logging.ERROR)

def sanitize_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', "-", str(name))

def get_wandb_run_name(args):
    base_model_name = sanitize_filename(args.base_model_name)

    return (
        f"{args.method}-"
        f"{args.resource}res-"
        f"{args.language}-"
        f"{base_model_name}-"
        f"{args.test_type}-"
        f"unseen-{args.unseen_author}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--language', type=str)
    parser.add_argument('--resource', type=str)
    
    parser.add_argument('--unseen_author', type=str, default=None)
    parser.add_argument('--test_type', type=str, default=None)
    parser.add_argument('--method', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    #parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--base_model_name', type=str, default="ai-forever/mGPT")
    
    parser.add_argument('--mask_filling_model_name',
                        type=str, default="t5-base")
    parser.add_argument('--cache_dir', type=str, default=".cache")
    args = parser.parse_args()
    
    assert args.test_type in ['ID', 'OOD']
    
    print(args)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("DEVICE: ", DEVICE)

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    #print(f'Loading dataset {args.config}...')
    #data, _ = load_dataset_mgt(config=args.config, test_type=args.test_type, unseen_author=args.unseen_author)
    data, _ = load_dataset_mgt(language=args.language, resource=args.resource, test_type=args.test_type, unseen_author=args.unseen_author)

    base_model_name = args.base_model_name.replace('/', '_')
    #SAVE_PATH = f"update_results/{base_model_name}-{args.mask_filling_model_name}/attribution_{args.method}_{args.config}_{args.test_type}"
    SAVE_PATH = f"update_results/{base_model_name}-{args.mask_filling_model_name}/attribution_{args.method}_{args.resource}_{args.language}_{args.test_type}"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_PATH)}")

    # write args to file
    with open(os.path.join(SAVE_PATH, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    wandb.login()
    wandb.init(project="multilingual-OOD-AA", name=get_wandb_run_name(args), config=args)

    # mask_filling_model_name = args.mask_filling_model_name
    # batch_size = args.batch_size
    # n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    # n_perturbation_rounds = args.n_perturbation_rounds
    # n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # get generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(
        args.base_model_name, cache_dir)
    load_base_model(base_model, DEVICE)

    def ll_criterion(text): return get_ll(
        text, base_model, base_tokenizer, DEVICE)

    def rank_criterion(text): return -get_rank(text,
                                               base_model, base_tokenizer, DEVICE, log=False)

    def logrank_criterion(text): return -get_rank(text,
                                                  base_model, base_tokenizer, DEVICE, log=True)

    def entropy_criterion(text): return get_entropy(
        text, base_model, base_tokenizer, DEVICE)

    def GLTR_criterion(text): return get_rank_GLTR(
        text, base_model, base_tokenizer, DEVICE)

    if args.method == "Log-Likelihood":
        output = run_threshold_experiment(
            data, ll_criterion, "likelihood", get_wandb_run_name(args))
    elif args.method == "Rank":
        output = run_threshold_experiment(data, rank_criterion, "rank", get_wandb_run_name(args))
    elif args.method == "Log-Rank":
        output = run_threshold_experiment(
            data, logrank_criterion, "log_rank", get_wandb_run_name(args))
    elif args.method == "Entropy":
        output = run_threshold_experiment(
            data, entropy_criterion, "entropy", get_wandb_run_name(args))
    elif args.method == "GLTR":
        output = run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR", get_wandb_run_name(args))
    # elif args.method == "OpenAI-D":
    #     outputs.append(
    #         run_supervised_experiment(
    #             data,
    #             model='roberta-base-openai-detector',
    #             cache_dir=cache_dir,
    #             batch_size=batch_size,
    #             DEVICE=DEVICE,
    #             finetune=True,
    #             num_labels=args.num_labels,
    #             epochs=args.epochs))
    # elif args.method == "ConDA":
    #     outputs.append(
    #         run_supervised_experiment(
    #             data,
    #             model='update_results/ConDA',
    #             cache_dir=cache_dir,
    #             batch_size=batch_size,
    #             DEVICE=DEVICE,
    #             finetune=True,
    #             num_labels=args.num_labels,
    #             epochs=args.epochs))
    # elif args.method == "ChatGPT-D":
    #     outputs.append(
    #         run_supervised_experiment(
    #             data,
    #             model='Hello-SimpleAI/chatgpt-detector-roberta',
    #             cache_dir=cache_dir,
    #             batch_size=batch_size,
    #             DEVICE=DEVICE,
    #             pos_bit=1,
    #             finetune=True,
    #             num_labels=args.num_labels,
    #             epochs=args.epochs))
    # elif args.method == "LM-D":
    #     outputs.append(
    #         run_supervised_experiment(
    #             data,
    #             model='distilbert-base-uncased',
    #             cache_dir=cache_dir,
    #             batch_size=batch_size,
    #             DEVICE=DEVICE,
    #             pos_bit=1,
    #             finetune=True,
    #             num_labels=args.num_labels,
    #             epochs=args.epochs,
    #             save_path=SAVE_PATH +
    #             f"/LM-D-{args.epochs}"))
    #
    # # run LRR
    # elif args.method == "LRR":
    #     outputs.append(run_perturbation_experiments(
    #         args, data, base_model, base_tokenizer, method="LRR"))
    #
    # # # run GPTZero: pleaze specify your gptzero_key in the args
    # elif args.method == "GPTZero":
    #     outputs.append(run_gptzero_experiment(data, api_key=args.gptzero_key))
    #
    # # run DetectGPT
    # elif args.method == "DetectGPT":
    #     outputs.append(run_perturbation_experiments(
    #         args, data, base_model, base_tokenizer, method="DetectGPT"))
    #
    # # run NPR
    # elif args.method == "NPR":
    #     outputs.append(run_perturbation_experiments(
    #         args, data, base_model, base_tokenizer, method="NPR"))

    wandb.log(output)
    wandb.run.summary.update(output)

    print("Finish")
