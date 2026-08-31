
import logging
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
)
import os
os.environ['HF_TOKEN'] = ""  # TODO: set your HF toke
import numpy as np
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def save_rank_entropy(args, input_ids, model, MODEL, CONTEXT_SIZE, novel):
    ranks = []
    entropy = []
    for idx in tqdm(range(1, len(input_ids[0])), desc="Computing ranks and entropy"):
        token = input_ids[0, idx]
        with torch.inference_mode():
            outputs = model(input_ids[:, max(idx - CONTEXT_SIZE, 0):idx])
        logits = outputs.logits[0, -1, :].float()
        probs = torch.softmax(logits, -1)

        curr_entropy = -(probs.clamp(1e-10) * torch.log2(probs.clamp(1e-10))).sum().item()
        entropy.append(curr_entropy)

        sorted_idx = torch.argsort(probs, descending=True)
        curr_rank = (sorted_idx == token).nonzero(as_tuple=True)[0].item()
        ranks.append(curr_rank)

    logging.info(f"Ranks len: {len(ranks)} and first 10 ranks: {ranks[:10]}\n\n")
    ranks = np.array(ranks, dtype=np.float32)
    np.savez_compressed(
        f"{args.save_path}/ranks/{MODEL}-context-{CONTEXT_SIZE}-llm-novel-{novel}-ranks.npz",
        ranks=ranks)

    logging.info(f"entropy len: {len(entropy)} and first 10 entropy: {entropy[:10]}\n\n")
    entropy = np.array(entropy, dtype=np.float32)
    np.savez_compressed(
        f"{args.save_path}/entropy/{MODEL}-context-{CONTEXT_SIZE}-llm-novel-{novel}-entropy.npz",
        entropy=entropy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='evaluator model')
    #parser.add_argument('--save_path', type=str, help='save path for token scores', default='./data')
    parser.add_argument('--save_path', type=str, help='save path for token scores', default='./TRACE-scores/')
    parser.add_argument('--books_location', type=str, help='file path for books (or texts)', default='./llm-texts-processed-trimmed/')  # TODO: 
    args = parser.parse_args()

    novels = sorted([f for f in os.listdir(f"{args.books_location}")
                     if f.endswith(".txt")])
    logging.info(f"Total number of novels: {len(novels)}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    CONTEXT_SIZE = 1024
    MODEL = args.model.replace('/', '-')
    logging.info(f"Eval Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model = torch.compile(model)
    model.eval()

    for novel in novels:
        logging.info(f"Novel: {novel}")
        with open(f"{args.books_location}/{novel}", "r", encoding="utf-8") as f:
            text = f.read()
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        logging.info("Tokens: %s", input_ids.shape)

        save_rank_entropy(args, input_ids, model, MODEL, CONTEXT_SIZE, novel)
        print("\n\n********************************************************\n\n")
