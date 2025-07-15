
import random
import pandas as pd
import json
from torch.utils.data import Dataset
import math

seed = 42
random.seed(seed)

data_path = 'model_responses_all/0_shot/'

demonstration = 1
num_demos=6

def _just_guess_prompt():
    
    pre_fix =  f'Provide your best guess for the following question. Give ONLY the guess, no other words or explanations, as short as possible; not a complete sentence, just the guess!\n\nThe question is: '

    return pre_fix


def _veb_prob_prompt():
    pre_fix = f"Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:\n\n"
    pre_fix += f"Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>\n"
    pre_fix += f"Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>"
    pre_fix += f"\n\nThe question is: "
    
    return pre_fix


def _cot_prompt():

    pre_fix = f"Briefly answer the following question by thinking step by step. Give the final answer (start with 'Answer: ' ) with minimal words in the end. \n\nThe question is: "

    return pre_fix



def load_dataset(dataname):
    
    dataset_path = data_path 

    ## load train data
    train = pd.read_csv(dataset_path + dataname+ "gemma2_0_shot_train.csv", encoding='utf-8')
    train_data = []
    for i in train.index.tolist():
        train_data.append({
             "question": train['question'][i],
             "answer": train['target'][i]
        })
    

    ## load demo data
    with open('data/' + dataname + "/train.txt", encoding='utf-8') as f:
        demo = f.read()
        demo = demo.split("\n\n")[:20]
    
    demo_data = []
    for item in demo:
        # print(item)
        if str(item.split("\n")[0]) not in [q['question'] for q in train_data]:
            demo_data.append({
                "question": str(item.split("\n")[0]),
                "answer": str(item.split("\n")[1])
            })

    demo_data = demo_data[:num_demos]

    ## load test data
    test = []
    with open(dataset_path + dataname+ "gemma2_0_shot_test.csv", encoding='utf-8') as f:
        for line in f:
            test.append(json.loads(line))

    test_data = []

    for item in test:
        test_data.append({
            "question": item['question'],
            "answer": item['target']
        })

    return train_data, test_data, demo_data


 
class QADataset(Dataset):
    def __init__(self, args, data, tokenizer):
        _, _, self.demo = load_dataset(args.dataname)
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        # self.device = device

    def __len__(self):
        # this should return the size of the dataset
        return len(self.data)
    
    def get_messages(self, idx):
        # this should return one sample from the dataset
        question = self.data[idx]['question'].strip()

        if self.args.prompt_type in ["few_shot"]:

            messages = []
            for item in self.demo:
                messages.append({"role": "user", "content": item['question']})
                messages.append({"role": "assistant", "content": item['answer']})

            messages.append({"role": "user", "content": question})

        else:
            if self.args.prompt_type in ["verb"]:
                pre_fix = _veb_prob_prompt()

            elif self.args.prompt_type in ['0_shot']:
                pre_fix = _just_guess_prompt()

            elif self.args.prompt_type in ['cot']:
                pre_fix = _cot_prompt()
            
            else:
                assert NotImplementedError
                
                
            messages = [{"role": "user", "content": pre_fix + question}]
  
        return messages
 
    def __getitem__(self, idx):
  
        question = self.data[idx]['question']
         
        if self.args.prompt_type in ["few_shot"]:
            messages = []
            for item in self.demo:
                messages.append({"role": "user", "content": item['question']})
                messages.append({"role": "assistant", "content": item['answer']})
            messages.append({"role": "user", "content": question})
            # print(messages)
            prompts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        

        else:
            if self.args.prompt_type in ["verb"]:

                pre_fix = _veb_prob_prompt()

            elif self.args.prompt_type in ['0_shot']:
                pre_fix = _just_guess_prompt()

            elif self.args.prompt_type in ['cot']:
                pre_fix = _cot_prompt()

            else:
                assert "no prompt type available!"
                
            messages = [{"role": "user", "content": pre_fix + question}]
            prompts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return prompts
    

class EvalDataset(Dataset):
    def __init__(self, response):
        
        self.q_id, self.questions, self.answers, self.targets =[], [], [], []
        for i, (q, a, t) in enumerate(zip(response['question'].tolist(), response['extracted_answer'].tolist(), response['target'].tolist())):
            ## collect multi target answers
            try:
                target_list = t.split('; ')
            except:
                target_list = [t]
            
            for targ in target_list:
                self.q_id.append(i)
                self.questions.append(q)
                self.answers.append(a)
                self.targets.append(targ)
        
    def __len__(self):
        # this should return the size of the dataset
        return len(self.questions)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        question = self.questions[idx]
        answer = self.answers[idx]
        target = self.targets[idx]
        q_id = self.q_id[idx]

        input_text = f"""###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 1, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 0 and 1. You should refer to the score rubric.
        3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 0 and 1)\"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {question}

        ###Response to evaluate:
        {answer}

        ###Reference Answer (Score 1):
        {target}

        ###Score Rubrics:
        Score 0: the response and reference answer to the instruction are not semantically equivalent.
        Score 1: the response and reference answer to the instruction are semantically equivalent.

        ###Feedback:  """
        

        return q_id, input_text


class CaliDataset(Dataset):
    def __init__(self, response, method):
        self.confidences, self.acces =[], [], [], []
        if method == "ts_infosel_confidence":
            self.confidences = response['infosel_logit'].tolist()
        elif method == "ts_seq_confidence":
            self.confidences = response['extracted_seq_prob'].tolist()
        
        else:
            assert "Not implemented method!"
        
        self.acces = response['extracted_prom_score'].tolist()

            
    def __len__(self):
        # this should return the size of the dataset
        return len(self.confidences)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        if math.isnan(self.confidences[idx]):
            confidence = 0
        else: 
            confidence = self.confidences[idx]
        acc = self.acces[idx]
    
        return confidence, acc
