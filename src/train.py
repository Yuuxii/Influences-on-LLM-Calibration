import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from cali_metrics import evaluate_metrics
import os
import pandas as pd
import random
# import json
torch.manual_seed(9595)
random.seed(9595)
np.random.seed(9595)
torch.cuda.manual_seed_all(9595)
import ast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## load question and answers from different LLMs
class MCEnsbDataset(Dataset):
    def __init__(self, args, train, tokenizer):
        response_path = 'model_responses_all/' + args.prompt_type + '/'
        self.models = args.include_models
        self.responses = {} 
        self.resp_accs = {}
        for model in args.include_models:
            if train == 'test':
                test_save_path = response_path + args.test_dataname +'_' + model +'_'+ args.prompt_type + "_test.csv"
                data = pd.read_csv(test_save_path, encoding='utf-8', index_col=False)
                
            else:
                data = pd.DataFrame(columns=['extracted_answer', 'extracted_prom46_score', 'question'])
                questions = []
                extracted_answers = []
                extracted_prom46_scores = []
                for i, train_dataname in enumerate(args.train_datanames):
                    train_save_path = response_path + train_dataname +'_' + model +'_'+ args.prompt_type + "_train.csv"
                    temp_data = pd.read_csv(train_save_path, encoding='utf-8', index_col=False)
                    questions.extend(temp_data['question'].tolist())
                    extracted_answers.extend(temp_data['extracted_answer'].tolist())
                    extracted_prom46_scores.extend(temp_data['extracted_prom46_score'].tolist())
                
                data['question'] = questions
                data['extracted_answer'] = extracted_answers
                data['extracted_prom46_score'] = extracted_prom46_scores
                # print(data)
                if train == 'data':  
                    data = data.iloc[0:round(data.shape[0]*0.8)]

                elif train == 'val':
                    data = data.iloc[round(data.shape[0]*0.8):]
                    data = data.reset_index(drop=True)

                # print(train_save_path)
            self.responses[model] = data['extracted_answer'].tolist()
            self.resp_accs[model] = data['extracted_prom46_score'].tolist()

        self.questions = data['question'].tolist()
        
        self.tokenizer = tokenizer

    def __len__(self):
        
        return len(self.questions)
    
    def __getitem__(self, index): 
   
        question = self.questions[index]
        anses = [self.responses[model][index] for model in self.models]
        ans_accs = [self.resp_accs[model][index] for model in self.models]

        num_ans = len(ans_accs)

        first_sentences = [[question] * num_ans ]
        second_sentences = [[f"{end}" for end in anses]]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        # print('error item: ', index, first_sentences, second_sentences)  
        tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True)

        return_item = {'question': question,
                       'label':    ans_accs,
                       }
        for idx, ans in enumerate(anses):
            return_item['ans'+str(idx)] = ans

        for k, v in tokenized_examples.items():
            for i in range(0, len(v), num_ans):
                return_item[k] = v[i : i + num_ans]
                
        return return_item
                

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, y_pred, y_true):
        bce_loss = nn.BCEWithLogitsLoss()(y_pred, y_true)
        pt = torch.exp(-bce_loss)  # pt is the probability of correct classification
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()
        return focal_loss
    

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        """
        y_pred: Predicted scores (logits) from the model, shape (batch_size,)
        y_true: Ground truth binary labels, shape (batch_size,)
        """
        # Get the indices of positive and negative samples
        pos_idx = (y_true == 1)
        neg_idx = (y_true == 0)
        
        # Predicted scores for positive and negative examples
        pos_pred = y_pred[pos_idx]  # Predictions for positive class
        neg_pred = y_pred[neg_idx]  # Predictions for negative class

        if len(pos_pred) == 0 or len(neg_pred) == 0:
            # If there's no positive or negative examples, return 0 loss
            return torch.tensor(0.0, requires_grad=True)

        # Pairwise differences between positive and negative predictions
        pairwise_diff = pos_pred.unsqueeze(1) - neg_pred.unsqueeze(0)  # Shape: (num_pos, num_neg)
        
        # RankNet loss (pairwise logistic loss)
        pairwise_loss = F.softplus(-pairwise_diff).mean()

        return pairwise_loss

class AUCSurrogateLoss(nn.Module):
    def __init__(self):
        super(AUCSurrogateLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        """
        y_pred: Predicted scores (logits) from the model, shape (batch_size,)
        y_true: Ground truth binary labels, shape (batch_size,)
        """
        pos_idx = (y_true == 1)
        neg_idx = (y_true == 0)
        
        pos_pred = y_pred[pos_idx]  # Predictions for positive class
        neg_pred = y_pred[neg_idx]  # Predictions for negative class

        # Pairwise difference between positive and negative predictions
        pairwise_diff = pos_pred.unsqueeze(1) - neg_pred.unsqueeze(0)  # Shape: (num_pos, num_neg)
        
        # Surrogate for AUC using a smooth approximation (sigmoid function)
        pairwise_loss = torch.sigmoid(-pairwise_diff).mean()

        return pairwise_loss
    
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        
        return batch


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        # print('input', inputs)
        outputs = model(**inputs)
        # print('output', outputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        if args.loss == 'bce':
            loss_fct = nn.BCEWithLogitsLoss()
        elif args.loss == 'focal':
            loss_fct = FocalLoss()
        elif args.loss == 'auc':
            loss_fct = AUCSurrogateLoss()
        elif args.loss == 'ranknet':
            loss_fct = RankNetLoss()
        else:
            assert 'please provide a loss!'

        loss = loss_fct(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = nn.Sigmoid()(torch.from_numpy(predictions))
    # predictions = np.argmax(predictions, axis=1)

    auc_scores = []
    ece_scores = []
    t_ece_scores = []
    brier_scores = []
    for idx, _ in enumerate(args.include_models):
        ensb_logits = predictions.T[idx]
        ensb_logits = [i.item() for i in ensb_logits]
        correctesses = labels.T[idx]
        eval_results = evaluate_metrics(correctesses, ensb_logits)
        auc_scores.append(eval_results['auc'])  
        ece_scores.append(eval_results['ece']) 
        t_ece_scores.append(eval_results['ece_temp_scaled']) 
        brier_scores.append(eval_results['brier_score']) 

    return {'ece': sum(ece_scores)/len(ece_scores), 
            't-ece': sum(t_ece_scores)/len(t_ece_scores),
            'brier': sum(brier_scores)/len(brier_scores),
            'auc':sum(auc_scores)/len(auc_scores)
            }

def return_print(em, prec, rec, f1):
  print('final_result:', args.use_amount)
  print(str(round(em*100, 2)) + ' & ' + str(round(prec*100, 2))+ ' & ' + str(round(rec*100, 2)) + ' & ' + str(round(f1*100, 2)))    

def test(trainer, test_dataset, test):
    print('---test---')
    preds, _, _= trainer.predict(test_dataset)

    preds = nn.Sigmoid()(torch.from_numpy(preds))
    
    response_path = 'model_responses_all/' + args.prompt_type + '/'
    result_path = 'ensb_results/' +  args.prompt_type + '/'
    
    for idx, model in enumerate(args.include_models):
        # print("before:", preds.T[idx])
        test_save_path = response_path + args.test_dataname +'_' + model +'_'+ args.prompt_type + "_" + test+ ".csv"
        response_data = pd.read_csv(test_save_path, encoding='utf-8', index_col=False)
        result_save_path = result_path + args.dataname +'_' + model +'_'+ args.prompt_type + "_" + test+ ".csv"
        
        
        if os.path.isfile(result_save_path):
            ensb_results = pd.read_csv(result_save_path, encoding='utf-8', index_col=False) 

        else:
            ensb_results = pd.DataFrame(columns=['extracted_prom46_score'])

        if test == 'train':
            ensb_logit = np.concatenate((np.zeros(response_data.shape[0]-len( preds.T[idx])), preds.T[idx]))
        else:
            ensb_logit = preds.T[idx]

        if args.prompt_type == 'vanilla':
            ensb_results['verb_prob'] = response_data['extracted_prob']
        
        ensb_results['llm_logit'] = response_data['extracted_seq_prob']
                
        ensb_results['extracted_prom46_score'] = response_data['extracted_prom46_score']
            
        ensb_results[args.loss + '_' + args.ensb_type + '_' + args.model + '_is_'+str(len(args.include_models))+ '_cs'] = ensb_logit
        # print("after:", [np.exp(p) for p in preds.T[idx]])

        
        ensb_results.to_csv(result_save_path, encoding='utf-8', index=False)
    
   

def load_model():
    if not args.evaluate:
        if args.use_pre:
            print('use pretrained model')
            model = AutoModelForMultipleChoice.from_pretrained("ncduy/bert-base-uncased-finetuned-swag")
            tokenizer = AutoTokenizer.from_pretrained("ncduy/bert-base-uncased-finetuned-swag")
        
        else:
                
            model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
    else:
        model = AutoModelForMultipleChoice.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)


    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true') 
    parser.add_argument('--use_pre',  action='store_true')    
    parser.add_argument('--include_models', nargs='+')
    parser.add_argument('--model', default='bert', help='choose from [bert, roberta, albert, deberta-v2]')   
    parser.add_argument('--output_dir', default='mc_bert/')  
    parser.add_argument('--dataname', default='sciq', help='choose from [sq, nq]')  
    parser.add_argument('--train_datanames', nargs='+', help='this setting is for generalization experiments')  
    parser.add_argument('--test_dataname', default='sciq', help='this setting is for generalization experiments, choose from test dataset')  
    parser.add_argument('--prompt_type', default='ans', help='choose prompt style')  
    parser.add_argument('--use_amount', default=1000, type=int)  
    parser.add_argument('--bs', default=16, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--opt', default='adamw')
    parser.add_argument('--test', action='store_true') # test on testdev data
    parser.add_argument('--ensb_type', default='small', help='choose what group of models you want to include, can be small, large and mix' )
    parser.add_argument('--loss', default='bce', help='bce, focal, auc')
    args = parser.parse_args()
    print('args:', args)


    model, tokenizer = load_model()
    
    ## data
    train_dataset = MCEnsbDataset(args, train='train',  tokenizer = tokenizer)
    eval_dataset = MCEnsbDataset(args, train='val', tokenizer = tokenizer)    
    test_dataset = MCEnsbDataset(args, train='test', tokenizer = tokenizer)  

    if not args.evaluate:
        
        print('train_dataset', len(train_dataset)) 

    print('test_dataset', len(test_dataset))  

    print('eval_dataset', len(eval_dataset))

    save_strategy = 'steps'
    load_best_model_at_end = True
    
    training_args = TrainingArguments(
        output_dir=args.output_dir +'/' + args.model + '/' + args.dataname + '_' + args.prompt_type,
        overwrite_output_dir = 'True',
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit = 2,
        seed=123,
        report_to='none'
        )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        )


    if not args.evaluate:
        print('start training!')

        trainer.train()
    
        test(trainer, test_dataset, 'test')

    else:
        test(trainer, test_dataset)

