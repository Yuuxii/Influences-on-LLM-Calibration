
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import torch

cache_dir = ''


def load_models(args):
   
    ## gemma-2b
    if args.model == 'gemma2':
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map='auto', cache_dir =  cache_dir)
      

    ## gemma-7b
    elif args.model == 'gemma9':
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map='auto', cache_dir =  cache_dir)
     

    ## mistral-7b
    elif args.model == 'mistral7':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",  device_map='auto', cache_dir =  cache_dir)


    ## mistral-8xb
    elif args.model == 'mixtral7':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir =  cache_dir).to(device)

    ## LLaMA-2-7b
    elif args.model == 'llama7':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",  device_map='auto', cache_dir =  cache_dir)

    elif args.model == 'llama31-8':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16,  device_map='auto', cache_dir =  cache_dir)
          
    ## LLaMA-3-8b
    elif args.model == 'llama8':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",  device_map='auto', cache_dir =  cache_dir)
        

    elif args.model == 'phi7':
        model_id = "microsoft/Phi-3-small-128k-instruct"
        model = AutoModelForCausalLM.from_pretrained(
                                                    model_id, 
                                                    trust_remote_code=True,  attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
                                                    torch_dtype=torch.bfloat16, device_map="auto", cache_dir =  cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    elif args.model == 'phi4':
        model_id = "microsoft/Phi-3.5-mini-instruct"
        model = AutoModelForCausalLM.from_pretrained(
                                                    model_id, 
                                                    trust_remote_code=True,  attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
                                                    torch_dtype=torch.bfloat16, device_map="auto", cache_dir =  cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        print(" model is not included! ")
    

    return model, tokenizer


def load_evaluate_models(model='prom46'):

    if model == "prom":
        tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
        model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-7b-v2.0", device_map="auto", cache_dir = cache_dir)
        
        return model, tokenizer

    elif model == "prom46":

        tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-8x7b-v2.0")
        model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-8x7b-v2.0", device_map="auto", torch_dtype=torch.bfloat16, cache_dir = "/data/lm_hosting/andi/cache/tgi_home/")

        return model, tokenizer
    
    elif model == "bleurt":
        bleurt = evaluate.load("bleurt", module_type="metric")
        return  bleurt
    
    else:
        print("This evaluation model is not included!")


