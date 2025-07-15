import multiprocessing
import os
import argparse

import pandas as pd
import numpy as np
import tqdm
from transformers import set_seed
from openai import OpenAI

from load_data_prompt import load_dataset, QADataset


def run_generation(idx, question, target, messages, model, url):

    if url is not None:
        client = OpenAI( base_url=url, api_key=" ")
    else:
        client = OpenAI()
    set_seed(123)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        logprobs=True,
        max_tokens=280,
        # stop=['<end_of_turn>', '<eos>', '<pad>', '<|end|>', '<|eot_id|>', '<|endoftext|>']
    )

    token_logprobs = np.array([t.logprob for t in response.choices[0].logprobs.content])
    token_probs = np.exp(token_logprobs)

    sentence_probs = np.exp(np.sum(token_logprobs) / len(token_logprobs))

    return {
        "question": question,
        "target": target,
        "response": response.choices[0].message.content,
        "ans_tok": [t.token for t in response.choices[0].logprobs.content],
        "token_probs": token_probs.tolist(),
        "seq_prob": float(sentence_probs),
    }


def generations_per_dataset(dataset, model, url):

    generations = []
    generation_inputs = []

    for idx in range(len(dataset.data)):

        generation_input = [
            idx,
            dataset.data[idx]["question"],
            dataset.data[idx]['answer'],
            dataset.get_messages(idx),
            model,
            url,
        ]
        generation_inputs.append(generation_input)

    num_processes = args.batch_size

    with multiprocessing.Pool(processes=num_processes) as pool:

        for res in tqdm.tqdm(
            pool.istarmap(run_generation, generation_inputs),
            total=len(generation_inputs),
        ):
            generations.append(res)

    generations_df = pd.DataFrame(generations)

    return generations_df


def main(response_path, args):

    # load raw data
    train_data, test_data, _ = load_dataset(args.dataname)
    dataset_train = QADataset(args, train_data, tokenizer=None, device=None)
    dataset_test = QADataset(args, test_data, tokenizer=None, device=None)

    train_save_path = (
        response_path
        + args.dataname
        + "_"
        + args.model.replace("/", "_")
        + "_"
        + args.prompt_type
        + "_train.csv"
    )
    print("train_save_path:", train_save_path)
    if not os.path.isfile(train_save_path):
        train_response = generations_per_dataset(dataset_train, args.model, url=args.url)
        train_response.to_csv(train_save_path, encoding="utf-8")

    test_save_path = (
        response_path
        + args.dataname
        + "_"
        + args.model.replace("/", "_")
        + "_"
        + args.prompt_type
        + "_test.csv"
    )
    if not os.path.isfile(test_save_path):
        test_response = generations_per_dataset(dataset_test, args.model, url=args.url)
        test_response.to_csv(test_save_path, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataname", default="truthfulqa", help="choose from [sq, truthfulqa]"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        help="choose from [gemma2, gemma7, zephyr7, mistral7, mixtral7, llama7, llama8, mpt7, falcon7, olmo7",
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--prompt_type", default="ans", help="choose from prompt types"
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--url", default="http://0.0.0.0:8001/v1", type=str)
    args = parser.parse_args()
    print("args:", args)

    response_path = "model_responses_openai/" + args.prompt_type + "/"

    os.makedirs(response_path, exist_ok=True)

    main(response_path, args)