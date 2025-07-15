from utils import evaluate_with_bleurt_score
from load_llms import load_evaluate_models
import argparse
import os
import pandas as pd
import torch
from load_data_prompt import EvalDataset
from transformers import set_seed
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

def extract_promp_score(response):
    promp_scores = [list(res.keys())[0] for res in response[args.eval_model+ '_score'].tolist()]
    promp_exps = [list(res.values())[0] for res in response[args.eval_model+ '_score'].tolist()]
    extracted_scores = []
    for score, exp, all in zip(promp_scores, promp_exps, response[args.eval_model+ '_score'].tolist()):
        if score == '-':
            if 'not' in exp.split('.')[0]:
                extracted_score = 0.
            else:
                extracted_score = 1.
        elif score == "'":
            extracted_score = float(all[2:5])

        else:
            try:
                extracted_score = float(score)
            except:
                extracted_score = float(-1)

        if extracted_score >= 0.5:
            extracted_score = 1.0
        else:
            extracted_score = 0.0

        extracted_scores.append(extracted_score)

    return extracted_scores 

def evaluate_with_prometheus_score(inputs):
   
    model.eval()

    tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids = tokenizer(inputs, return_tensors="pt", padding=True).to(model.device)
 
    prompt_length = input_ids['input_ids'].shape[1]
    set_seed(123)
    outputs = model.generate(**input_ids, max_new_tokens=300, repetition_penalty=1.03)
    eval_results = tokenizer.batch_decode(outputs[:,prompt_length:], skip_special_tokens=True)

    extracted_scores = []
    extracted_exps = []
    for result in eval_results:
        try:
            score = result.strip().split('[RESULT] ')[1]
        except:
            score = -1
            
        explanation = result
        try:
            numerical_score = int(score)
        except:
            numerical_score = score

        extracted_scores.append(numerical_score)
        extracted_exps.append(explanation)

    return extracted_scores, extracted_exps

def get_evaluate_results(dataloader):

    ## get bleurt score
    all_q_id, all_prom_score, all_prom_exp = [], [], []
    for idx, (q_ids, inputs) in enumerate(dataloader):
     
        prom_scores, prom_exps = evaluate_with_prometheus_score(inputs) 
        all_q_id.extend(q_ids)
        all_prom_score.extend(prom_scores)
        all_prom_exp.extend(prom_exps)
 
    all_results = dict()
    for q_id in all_q_id:
        all_results[q_id.item()] = []

    for q_id, prom_score, prom_exp in zip(all_q_id, all_prom_score, all_prom_exp): 
        
        all_results[q_id.item()].append({
            args.eval_model+ '_score': prom_score,
            'prom_exp': prom_exp
        })
    
    prom_result_per_question = []
    for i in range(len(all_results.keys())):
        max_prom_score = -1
        max_prom_explanation = 'None'
     
        for item in all_results[i]:
            prom_score = item[args.eval_model+ '_score']
            prom_exp = item['prom_exp']

            if isinstance(prom_score, int):
                if isinstance(max_prom_score, int):
                    if prom_score > max_prom_score:
                        max_prom_score = prom_score
                        max_prom_explanation = prom_exp
                    elif max_prom_score == -1:
                        max_prom_explanation = prom_exp
                else:
                    max_prom_score = prom_score
                    max_prom_explanation = prom_exp

            else:
                max_prom_score = prom_score
                max_prom_explanation = prom_exp

        prom_result_per_question.append({max_prom_score: max_prom_explanation})
    
    return prom_result_per_question
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='truthfulqa', help='choose from [sq, truthfulqa]')    
    parser.add_argument('--model', default='gemma7', help='choose from [gemma2, gemma7, zephyr7, mistral7, mixtral7, llama7, llama8, mpt7, falcon7, olmo7') 
    parser.add_argument('--eval_model', default='prom46')
    parser.add_argument('--evaluate', default='all')
    parser.add_argument('--prompt_type', default='ans', help='choose from [topk, ans, vanilla, cot]' )
    parser.add_argument('--batch_size', default=20, type=int)  
    args = parser.parse_args()
    print('args:', args)

    response_path = 'model_responses_' + args.evaluate + '/' + args.prompt_type + '/'

    
    train_save_path = response_path + args.dataname +'_' + args.model +'_'+ args.prompt_type + "_train.csv"
    test_save_path = response_path + args.dataname +'_' + args.model +'_'+ args.prompt_type + "_test.csv"
        
    if os.path.isfile(train_save_path):
        
        train_response = pd.read_csv(train_save_path, encoding='utf-8')
       
        model, tokenizer = load_evaluate_models(model=args.eval_model)
        
        
        dataset_train = EvalDataset(train_response)

        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            drop_last=False,
        )
       
        train_response[args.eval_model + '_score'] = get_evaluate_results(train_dataloader)

        extracted_prom_score = extract_promp_score(train_response)
        train_response['extracted_' + args.eval_model + '_score'] = extracted_prom_score
     
        train_response.to_csv(train_save_path, encoding ='utf-8', index=False)
        print('saved results to: ', train_save_path)
            

    if os.path.isfile(test_save_path):
      
        test_response = pd.read_csv(test_save_path, encoding='utf-8')
        dataset_test = EvalDataset(test_response)
      
    
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            drop_last=False,
        )
    
        test_response[args.eval_model + '_score'] = get_evaluate_results(test_dataloader)

        extracted_prom_score = extract_promp_score(test_response)
        test_response['extracted_' + args.eval_model + '_score'] = extracted_prom_score


        test_response.to_csv(test_save_path, encoding ='utf-8', index=False)

