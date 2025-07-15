import torch
from transformers import set_seed
import datasets
from load_data_prompt import load_dataset, QADataset
import argparse
import os
import pandas as pd
from load_llms import load_models
import numpy as np

cache_dir = ''
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
debug = 0

def get_verb_response(inputs):
    model.eval()
    if args.prompt_type == "verb":
        max_new_tokens = 100
    else:
        max_new_tokens = 280

    tokenizer.pad_token_id = model.config.eos_token_id
    
    set_seed(123)
    if args.model == 'llama31-8':
        tokenizer.pad_token = "<|eot_id|>"
        tokenizer.padding_side = 'left'
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature = 1, 
                                eos_token_id=tokenizer.eos_token_id)

    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[:, input_length:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


    return responses


def get_model_response(inputs):

    model.eval()
    if args.prompt_type == 'cot':
        max_new_tokens=300
    else:
        max_new_tokens=100 


    set_seed(123)
    if debug:
        print(inputs)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature = 1, 
                                output_scores=True, return_dict_in_generate=True, 
                                )
    
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs.sequences[:, input_length:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )

    seq_scores = []
    all_tokens = []
    all_tok_scores = [] 
    for generated_id, scores_of_label in zip(generated_ids, transition_scores):
    # print(scores_of_label)
        filted_scores_of_label = [s for s in scores_of_label if np.exp(s.item()) != 0.0 or np.exp(s.item()) != 0 ]
        seq_scores.append(torch.exp(sum(filted_scores_of_label)/len(filted_scores_of_label)).item())
        tokens = []
        probs = []
        for tok, score in zip(generated_id, scores_of_label):
            tokens.append(tokenizer.decode(tok))
            probs.append(np.exp(score.item()))

        all_tokens.append(tokens)
        all_tok_scores.append(probs)


    if debug:
        print('respones:', responses)

    return responses, all_tokens, all_tok_scores, seq_scores



def get_dataset_responses(args, data, model_predictions, save_path):

    all_seq_scores = []
    all_responses = []
    all_tokens = []
    all_tok_scores = []
    for idx, inputs in enumerate(data):
  
        responses, batch_tokens, batch_tok_scores, seq_scores = get_model_response(inputs)
        all_tokens.extend(batch_tokens)
        all_tok_scores.extend(batch_tok_scores)
        all_seq_scores.extend(seq_scores)
        all_responses.extend(responses)
          
    model_predictions['ans_tok'] = all_tokens
    model_predictions['tok_prob'] = all_tok_scores
    model_predictions['seq_prob'] = all_seq_scores
    model_predictions['response'] = all_responses

    model_predictions.to_csv(save_path, encoding ='utf-8')


def main(args):
    
    ## load raw data
    train_data, test_data, _ = load_dataset(args.dataname)
    dataset_train = QADataset(args, train_data, tokenizer, device=model.device)
    dataset_test = QADataset(args, test_data, tokenizer, device=model.device)

    # Create DataLoaders for the training and validation dataset

    train_save_path = response_path + args.dataname +'_' + args.model +'_'+ args.prompt_type + "_train.csv"
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        drop_last=False,
    )
    if not os.path.isfile(train_save_path):
        train_response = pd.DataFrame(columns=['question', 'target', 'response', 'ans_tok', 'tok_prob', 'seq_prob'])
        train_response['question'] = [item['question'] for item in train_data]
        train_response['target'] = [item['answer'] for item in train_data]
    
    else:
        train_response = pd.read_csv(train_save_path, encoding='utf-8', index_col=False)
        
    get_dataset_responses(args, train_dataloader, train_response, train_save_path)

    
    test_save_path = response_path + args.dataname +'_' + args.model +'_'+ args.prompt_type + "_test.csv"
    test_dataloader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    drop_last=False,
    )
    if not os.path.isfile(test_save_path):
        test_response = pd.DataFrame(columns=['question', 'target', 'response', 'ans_tok', 'tok_prob', 'seq_prob'])
        test_response['question'] = [item['question'] for item in test_data]
        test_response['target'] = [item['answer'] for item in test_data]
    else:
        test_response = pd.read_csv(test_save_path, encoding='utf-8', index_col=False)
    
    get_dataset_responses(args, test_dataloader, test_response, test_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='truthfulqa')    
    parser.add_argument('--model', default='gemma7', help='choose a model to generate answers') 
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--prompt_type', default='cot' )
    parser.add_argument('--batch_size', default=64, type=int)  
    args = parser.parse_args()
    print('args:', args)

    response_path = 'model_responses_all/' + args.prompt_type + '/'

    if not os.path.isdir(response_path):
        os.mkdir(response_path)
    ## load model
    model, tokenizer = load_models(args)
    
    main(args)