from parsing import parse_single_guess_safe
import argparse
import os
import pandas as pd
import re
import numpy as np
import ast
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

# Compile a regular expression pattern to match any of the custom punctuation
punctuation_pattern = re.compile(f"[{re.escape("!#*?/_")}]")

explanation_start = ["Here's a little more to help understand",
                     "Here's why",
                     "Here's the breakdown",
                     "Here's a breakdown",
                     "Here's how it works",
                     "Here are some reasons why",
                     "Here's the truth"]

def longest_ones_indices(lst):
    max_length = 0
    max_start = -1
    current_length = 0
    current_start = -1

    # Iterate through the list to find the longest sequence of 1s
    for i, value in enumerate(lst):
        if value == 1:
            if current_length == 0:
                current_start = i  # Start of a new sequence of 1s
            current_length += 1
        else:
            # Check if the current sequence is the longest so far
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
            current_length = 0  # Reset the count for the next sequence

    # Final check for the last sequence
    if current_length > max_length:
        max_length = current_length
        max_start = current_start

    # Return all indices of the longest sequence of 1s
    if max_length > 0:
        return list(range(max_start, max_start + max_length))
    else:
        return [] 
    

def remove_common(response):
    response = str(response)
    response =  emoji_pattern.sub(r'', response) # no emoji
    response = punctuation_pattern.sub(r'', response)  # no unecessary punctuations
    response = response.replace('assistant', '')
    response = response.replace('...', '').replace('?', '')

    return response

def extract_verb_response(response):
    
    response = remove_common(response)

    guess, prob = parse_single_guess_safe(response)
    try:
        if guess[0] == "FAILED TO PARSE":
            print(response[0]['generated_text'][len(response):])
            return [response[0]['generated_text'][len(response):]], prob
        else:
            return guess.replace('Guess:', ''), prob
    except:
        return guess.replace('Guess:', ''), prob
    

def extract_few_shot_response(response):

    response = remove_common(response)

    ## break response to paragraphs
    extracted_answer = response.split('\n\n\n')[0]

    for exp_start in explanation_start:
        if len(extracted_answer.split(exp_start)[0].strip()) < 1:
            extracted_answer = extracted_answer
        else:
            extracted_answer = extracted_answer.split(exp_start)[0]

    extracted_answer = extracted_answer.strip().replace('assistant', '').replace('answer:', '').replace('answers:', '')

    return extracted_answer.strip()

def extract_0_shot_responses(response):
    # response = response.split('\n\n\n')[0]
    response = remove_common(response)
    response = response.strip().replace('assistant', '').replace('answer:', '').replace('answers:', '')
    
    response = response.split('Guess:')[-1]
   

    return response.strip()

def extract_cot_responses(response):
    
    response = remove_common(response)
    response = response.strip().replace('assistant', '').replace('answer:', '').replace('answers:', '')

    response = response.split('Answer:')[-1].split('\n\n')[0]

    return response.strip()


def extracted_llm_logit(extracted_ans, ans_toks, tok_probs):

    ## need to fisrt clean the commas.
    cleaned_toks = []
    cleaned_probs = []
    ans_toks = ast.literal_eval(ans_toks)
    tok_probs = ast.literal_eval(tok_probs)
    assert len(ans_toks) == len(tok_probs), "sizes of ans_toks and tok_probs are not match! " 

    for idx, (tok, prob) in enumerate(zip(ans_toks, tok_probs)):
        if tok.strip() not in [':', '...', '*', '#', '\n', '?', '_', '/', "assistant",'answer:', 'answers:', 'Answer:', '']:
            cleaned_toks.append(tok.strip())
            cleaned_probs.append(prob)

    toks_in_response = [1 if c_tok in extracted_ans else 0 for c_tok in cleaned_toks]
    response_idx = longest_ones_indices(toks_in_response)
    
    if len(response_idx) == 1:
    
        extracted_ans = extracted_ans.strip()
        response_toks = [extracted_ans]
            
        try:
            response_probs = [cleaned_probs[cleaned_toks.index(extracted_ans)]]
        except:
            ## in case the extracted_ans is a nan value and can not be found in the cleaned_toks list
            response_probs = cleaned_probs

    else:
        response_probs = [cleaned_probs[idx] for idx in response_idx]
        response_toks = [cleaned_toks[idx] for idx in response_idx]
    # print(response_probs)
    extracted_response_logit = np.log(np.array(response_probs)).sum()/len(response_probs)

    return response_toks, np.exp(extracted_response_logit)

def extract_responses(responses):

    extracted_anses = []
    extracted_probs = []
    extracted_ans_toks = []
    extracted_seq_probs = []
    if args.evaluate == 'openai':
        tok_prob = "token_probs"
    else:
        tok_prob = 'tok_prob'
    for response, ans_tok, ans_prob  in zip(responses['response'].tolist(), responses['ans_tok'].tolist(), responses[tok_prob].tolist()):

        if args.prompt_type == 'few_shot':
            extracted_ans = extract_few_shot_response(response)
            extracted_anses.append(extracted_ans)
            extracted_ans_tok, extracted_seq_prob = extracted_llm_logit(extracted_ans, ans_tok, ans_prob)
            extracted_ans_toks.append(extracted_ans_tok)
            extracted_seq_probs.append(extracted_seq_prob)

        elif args.prompt_type == '0_shot':
            extracted_ans = extract_0_shot_responses(response)
            extracted_anses.append(extracted_ans)
            extracted_ans_tok, extracted_seq_prob = extracted_llm_logit(extracted_ans, ans_tok, ans_prob)
            extracted_ans_toks.append(extracted_ans_tok)
            extracted_seq_probs.append(extracted_seq_prob)
        
        elif args.prompt_type == 'cot':
            extracted_ans = extract_cot_responses(response)
            extracted_anses.append(extracted_ans)
            extracted_ans_tok, extracted_seq_prob = extracted_llm_logit(extracted_ans, ans_tok, ans_prob)
            extracted_ans_toks.append(extracted_ans_tok)
            extracted_seq_probs.append(extracted_seq_prob)

        elif args.prompt_type == 'verb':
            extracted_ans, verb_prob = extract_verb_response(response)
            extracted_anses.append(extracted_ans)
            extracted_probs.append(verb_prob)
            extracted_ans_tok, extracted_seq_prob = extracted_llm_logit(extracted_ans, ans_tok, ans_prob)
            extracted_ans_toks.append(extracted_ans_tok)
            extracted_seq_probs.append(extracted_seq_prob)
        
        else:
            assert NotImplementedError

            
    responses['extracted_answer'] = extracted_anses
    responses['extracted_ans_tok'] = extracted_ans_toks
    responses['extracted_seq_prob'] = extracted_seq_probs

    if len(extracted_probs) != 0:
        responses['extracted_prob'] = extracted_probs
   
    return responses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='truthfulqa', help='choose from [sq, truthfulqa]')    
    parser.add_argument('--model', default='gemma7', help='choose from [gemma2, gemma7, zephyr7, mistral7, mixtral7, llama7, llama8, mpt7, falcon7, olmo7') 
    parser.add_argument('--evaluate', default='all',)
    parser.add_argument('--prompt_type', default='ans', help='choose from [topk, ans, verb, cot]' )
    # parser.add_argument('--num_gpus', default=1, type=int)  
    args = parser.parse_args()
    print('args:', args)
    
    response_path = 'model_responses_' + args.evaluate + '/' + args.prompt_type + '/'

    train_save_path = response_path + args.dataname +'_' + args.model +'_'+ args.prompt_type + "_train.csv"
    test_save_path = response_path + args.dataname +'_' + args.model +'_'+ args.prompt_type + "_test.csv"
    print(test_save_path)

    if os.path.isfile(train_save_path):
        train_response = pd.read_csv(train_save_path, encoding='utf-8')
        train_response = extract_responses(train_response)
        
        train_response.to_csv(train_save_path, encoding ='utf-8', index=False)


    if os.path.isfile(test_save_path):
       
        test_response = pd.read_csv(test_save_path, encoding='utf-8')
        test_response = extract_responses(test_response)
        test_response.to_csv(test_save_path, encoding ='utf-8', index=False)
