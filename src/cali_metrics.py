
import numpy as np
from plot import plot_reliability_diagram
from sklearn.metrics import roc_auc_score
from scipy.stats import sem 
import pandas as pd
import os
import math
import matplotlib as plt

CONFIDENCE_EXPRESSIONS = ['Almost No Chance', 'Highly Unlikely', 'Chances are Slight', 'Little Chance', 
                          'Unlikely', 'Probably Not', 'About Even', 'Better than Even', 'Likely',
                          'Probably', 'Very Good Chance', 'Highly Likely', 'Almost Certain']

CONFIDENCE_EXPRESSIONS_PROBABILITIES = [0.02,0.05,0.1,0.1,0.2,0.25,0.5,0.6,0.7,0.7,0.8,0.9,0.95]

EPS = 1e-8

def compute_ece(accuracies, confidences, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0
    for idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Calculated |confidence - accuracy| in each bin
        if idx != n_bins - 1:
            in_bin = (confidences >= bin_lower) * (confidences < (bin_upper))
        else:
            in_bin = (confidences >= bin_lower) * (confidences <= (bin_upper))
        prop_in_bin = in_bin.mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def logp(probs):
    return np.log(np.clip(probs, EPS, 1 - EPS))

def find_temperature(probs, labels):
    logps = logp(probs)
    temperatures = 10 ** np.linspace(-1.25, 2, 10000)
    scaled_probs = np.exp(logps.reshape(1, -1) / temperatures.reshape(-1, 1))

    # compute the loss for each temperature
    losses = nll(scaled_probs, labels)

    # find the temperature that minimizes the loss
    best_temperature = temperatures[np.argmin(losses)]

    return best_temperature

def calibrate_with_temperature(probs, temperature):
    return np.exp(logp(probs) / temperature)

def nll(probs, labels):
    pos_mask = labels.reshape(1, -1).astype(bool)
    neg_mask = ~pos_mask
    losses = -np.log(pos_mask * probs + neg_mask * (1 - probs) + EPS).sum(axis=-1) / len(probs)
    return losses


def auroc(accuracies, confidences, ax=None):

    return roc_auc_score(accuracies, confidences)

def compute_brier_score(corrects, confidences):
    corrects = np.array(corrects)
    confidences = np.array(confidences)
    return np.mean((corrects - confidences)**2)


# return dictionary of all evaluation metrics
def evaluate_metrics(accuracies, confidences, temp=None, cov_thresh_list=None, acc_thresh_list=None):
    if cov_thresh_list is None:
        cov_thresh_list = [50]
    if acc_thresh_list is None:
        acc_thresh_list = [50, 70, 90]

    failed_verbs = 0
    filted_accuracies = []
    filted_confidences = []
    for acc, conf in zip(accuracies, confidences):
        # print(conf)
        if conf < 0 or math.isnan(conf):
            # print(1)
            failed_verbs += 1
        else:
            filted_accuracies.append(acc)
            filted_confidences.append(conf)
    
    # if temperature is not provided, just fit on the same data 
    if temp is None:
        temp = find_temperature(np.array(filted_confidences), np.array(filted_accuracies))
    
    # compute dict of metrics to return 
    metrics_json = {}

    metrics_json['success'] = 1 - (failed_verbs/len(accuracies))
    # print(failed_verbs)
    metrics_json['temp'] = temp 
    
    metrics_json['accuracy'] = sum(filted_accuracies)/len(filted_accuracies)
    metrics_json['ece'] = compute_ece(np.array(filted_accuracies), np.array(filted_confidences))
    metrics_json['brier_score'] = compute_brier_score(filted_accuracies, filted_confidences)
    # temperature scaled ece 
    confidences_scaled = calibrate_with_temperature(filted_confidences, temp)
    # metrics_json['confidences_scaled'] = confidences_scaled
    metrics_json['brier_score_scaled'] = compute_brier_score(filted_accuracies, confidences_scaled)
    metrics_json['ece_temp_scaled'] = compute_ece(np.array(filted_accuracies), np.array(confidences_scaled))
    metrics_json['auc'] = auroc(filted_accuracies, filted_confidences)

    return metrics_json


if __name__ == '__main__':

    plot = 0
    avg = 0
    acc_sucess = 0 
    every_model = 0
    datanames = [ 'triviaqa', 'sciq', 'wikiqa', 'nq' ]
    ensb_type = 'small'
    models = ['llama7', 'llama8', 'llama31-8', 'gemma2', 'gemma9', 'phi4', 'phi7']
    methods = ['llm_logit', 'ts_seq_confidence', 'verb_prob', 'apricot_cluster_confidences', 
               'auc_one_bert_is_1',  'auc_small_bert_is_5', 
               'bce_one_bert_is_1', 'bce_small_bert_is_5',  'ts_infosel', 
               'focal_one_bert_is_1', 'focal_small_bert_is_5', 'ts_infosel_focal', 
               'auc_one_deberta_is_1', 'auc_small_deberta_is_5', 
               'bce_one_deberta_is_1', 'bce_small_deberta_is_5', 'ts_infosel_deberta',
                 'focal_one_deberta_is_1', 'focal_small_deberta_is_5', 'ts_infosel_focal_deberta']

    prompt_types = ['verb', '0_shot',  'cot', 'few_shot']
    
    eval_result_path = "result_analysis/eval_results/"

    acc_success_metrics = pd.DataFrame(columns=['model', 'prompt_type', 'triviaqa_success', 'triviaqa_acc',  'sciq_success', 'sciq_acc',  'wikiqa_success', 'wikiqa_acc', 'nq_success', 'nq_acc'])
    accs, success = {}, {}
    for dataname in datanames:
        accs[dataname] = []
        success[dataname] = []

    avg_cali_metrics = pd.DataFrame(columns=['model', 'method'])
    avg_eces, avg_t_eces, avg_briers, avg_aucs = {}, {}, {}, {}
    for prompt_type in prompt_types:  
        avg_eces[prompt_type] = []
        avg_t_eces[prompt_type] = []
        avg_briers[prompt_type] = []
        avg_aucs[prompt_type] = []
    
    all_cali_metrics = pd.DataFrame([])
    for model in models:
        cali_metrics = pd.DataFrame(columns=['model', 'prompt_type', 'method'])
        eces, t_eces, briers, aucs = {}, {}, {}, {}

        for dataname in datanames:
                eces[dataname] = []
                t_eces[dataname] = []
                briers[dataname] = []
                aucs[dataname] = []
        
        for dataname in datanames:
         
            for prompt_type in prompt_types:
                print(model, prompt_type, dataname)
                response_path = 'results/' + prompt_type + '/'
                test_save_path = response_path + dataname +'_' + model +'_'+ prompt_type + "_test.csv"
         
                if os.path.isfile(test_save_path):
                    test_response = pd.read_csv(test_save_path, encoding='utf-8')
        
                    accuracies = test_response['extracted_prom46_score'].tolist()
                    for method in methods:

                        if method == 'verb_prob' and prompt_type != 'verb':
                            eces[dataname].append(-1)
                            t_eces[dataname].append(-1)
                            briers[dataname].append(-1)
                            aucs[dataname].append(-1)
                        else:  
                            eval_results = evaluate_metrics(accuracies, test_response[method].tolist())
                            eces[dataname].append(round(eval_results['ece'], 3))
                            t_eces[dataname].append(round(eval_results['ece_temp_scaled'], 3))
                            briers[dataname].append(round(eval_results['brier_score'], 3))
                            aucs[dataname].append(round(eval_results['auc'], 3))

                            if plot:
                            
                                plot_reliability_diagram(test_response[method].tolist(), accuracies, num_bins=10, save_path = 'plots/' + model + '_' + prompt_type + '_' +dataname + '_'+ method+ '.pdf')    
                                
                                plt.pyplot.close()
                        
                    success[dataname].append(round(evaluate_metrics(accuracies, test_response[methods[0]].tolist())['success']*100, 2))
                    accs[dataname].append(round(evaluate_metrics(accuracies, test_response[methods[0]].tolist())['accuracy']*100, 2))


            cali_metrics[dataname+'_ece']=eces[dataname]
            cali_metrics[dataname+'_t-ece']=t_eces[dataname]
            cali_metrics[dataname+'_brier']=briers[dataname]
            cali_metrics[dataname+'_auc']=aucs[dataname]

        
        for i in range(cali_metrics.shape[0]): 
            
            avg_ece = sum([eces[dataname][i] for dataname in datanames])/len(datanames)
            avg_t_ece = sum([t_eces[dataname][i] for dataname in datanames])/len(datanames)
            avg_brier = sum([briers[dataname][i] for dataname in datanames])/len(datanames)
            avg_auc = sum([aucs[dataname][i] for dataname in datanames])/len(datanames)
            prompt_type_idx = i // len(methods)
            avg_eces[prompt_types[prompt_type_idx]].append(avg_ece)
            avg_t_eces[prompt_types[prompt_type_idx]].append(avg_t_ece)
            avg_briers[prompt_types[prompt_type_idx]].append(avg_brier)
            avg_aucs[prompt_types[prompt_type_idx]].append(avg_auc)


        prompt_type_for_metric = []
        method_for_metric = []
        for prompt_type in prompt_types:
            for method in methods:
                prompt_type_for_metric.append(prompt_type)
                method_for_metric.append(method)

        cali_metrics['method'] = method_for_metric
        cali_metrics['prompt_type'] = prompt_type_for_metric
        cali_metrics['model'] = [model for i in range(len(method_for_metric))]
        if every_model:
            cali_metrics.to_csv(eval_result_path + model  + '_'+ ensb_type + str(len(models)) + '.csv', encoding='utf-8', float_format='%11.3f')
        else:
            all_cali_metrics = pd.concat([all_cali_metrics, cali_metrics])
            all_cali_metrics.to_csv(eval_result_path + datanames[0] +'_' + ensb_type + str(len(models)) + '.csv', encoding='utf-8', float_format='%11.3f')

    if acc_sucess:
        model_for_metric = []
        prompt_type_for_metric = []
        for model in models:
            for prompt_type in prompt_types:
                model_for_metric.append(model)
                prompt_type_for_metric.append(prompt_type)

        acc_success_metrics['model'] = model_for_metric
        acc_success_metrics['prompt_type'] = prompt_type_for_metric
    
        for dataname in datanames:
            
            acc_success_metrics[dataname+'_acc'] = accs[dataname]
            acc_success_metrics[dataname+'_success'] = success[dataname]

        acc_success_metrics.to_csv(eval_result_path + 'acc_success_small.csv', encoding='utf-8', float_format='%11.3f')

    if avg:
        for prompt_type in prompt_types:

            avg_cali_metrics[prompt_type + '_avg_ece'] = avg_eces[prompt_type]
            avg_cali_metrics[prompt_type + '_avg_t-ece'] = avg_t_eces[prompt_type]
            avg_cali_metrics[prompt_type + '_avg_brier'] = avg_briers[prompt_type]
            avg_cali_metrics[prompt_type + '_avg_auc'] = avg_aucs[prompt_type]

        model_for_avg_cali_metric = []
        method_for_avg_cali_metric = []
        for model in models:
            for method in methods:
                model_for_avg_cali_metric.append(model)
                method_for_avg_cali_metric.append(method)

        avg_cali_metrics['model'] =  model_for_avg_cali_metric
        avg_cali_metrics['method'] = method_for_avg_cali_metric

        avg_cali_metrics.to_csv(eval_result_path + 'avg_cali_' + ensb_type + str(len(models)) + '.csv', encoding='utf-8', float_format='%11.3f')

    