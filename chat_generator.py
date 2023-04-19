import os, json, itertools, bisect, gc

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from accelerate import Accelerator
import accelerate
import time

model = None
tokenizer = None
generator = None
os.environ["CUDA_VISIBLE_DEVICES"]="0"

flag = "ExplanationGenerator" ## "Qiming-Alpaca", "ExplanationGenerator", "ExplanationVerifier"
if flag == "Qiming-Alpaca":
    load_model_name = "./qiming_alpaca/"
    First_chat = "Qiming-Alpaca: I am Qiming-Alpaca, what questions do you have?"
    invitation = "Qiming-Alpaca: "
    human_invitation = "User: "
elif flag == "ExplanationGenerator":
    # load_model_name = "./llama_7B_hf/llama-7b/"
    # load_model_name = "./qiming_llama_7B_Cardiff_Sydney_merged_generator/"
    # load_model_name = "./qiming_alpaca_7B_Cardiff_Sydney_merged_generator/"
    # load_model_name = "./qiming_alpaca_7B_Cardiff_generator/"
    # load_model_name = "./LLaMA_7B_Cardiff_generator/"
    load_model_name = "./qiming_alpaca_7B/"
    First_chat = "Explanation Generator: I am an expert in explantion generator, what questions can I help?"
    invitation = " Output: "
    human_invitation = "User: "
    
def load_model(model_name, eight_bit=0, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        #device_map=device_map,
        #device_map="auto",
        torch_dtype=torch.float16,
        #max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
        #load_in_8bit=eight_bit,
        #from_tf=True,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()

    generator = model.generate

load_model(load_model_name)

def explanationGenerator(history, global_step, msg, global_score_tag, response):
    # invitation = "Qiming-Alpaca: "
    # human_invitation = "User: "

    # input
    if global_step == 0:
        history.append(msg)
        fulltext = "Instruction: As an explanation generation expert, can you generate the explanation for the given input? \n\n" + \
        "\n\n".join(history) + "\n\n" + invitation
    else:
        #fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + "\n\n".join(history) + "\n\n" + invitation
        # fulltext = "\n\n".join(history) + "\n\n" + invitation
        fulltext = "Instruction: Your last time verifier rating score for your explanation generation is " + str(global_score_tag) + " and the explanation you generated was " + response + " As a explanation generation expert, can you generate a better explanation for the given input? \n\n" + \
            "\n\n".join(history) + "\n\n" + invitation        
    
    #print('SENDING==========')
    #print(fulltext)
    #print('==========')

    generated_text = ""
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
    in_tokens = len(gen_in)
    with torch.no_grad():
            generated_ids = generator(
                gen_in,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5, # default: 1.0
                top_k = 50, # default: 50
                top_p = 1.0, # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

            text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt

    response = response.split(human_invitation)[0]

    response = response.strip()

    # print(invitation + response)

    # print("")

    # history.append(invitation + response)
    
    return response, history

