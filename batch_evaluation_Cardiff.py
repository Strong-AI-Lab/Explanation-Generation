## The steps about data preprocessing for explanation generator
## Step 1: remove the question, options which is nan and including '<img' tag and the total rating lower than 10.
## Step 2: clean the question, options and explantion where has the html tag and extra white space.
## Step 3: remove the the explanation where there is nothing after cleaning the explanation.

import pandas as pd
import json
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm
# load_model_name = "./llama_7B_hf/llama-7b/"
# load_model_name = "./qiming_llama_7B_Cardiff_Sydney_merged_generator/"
# load_model_name = "./qiming_alpaca_7B_Cardiff_Sydney_merged_generator/"
# load_model_name = "./qiming_alpaca_7B/"
load_model_name = "./LLaMA_7B_Cardiff_generator/"

# load_model_name = "./qiming_alpaca_7B_Cardiff_generator/"
path_list = ["./Paul_new_data/"]
# cardiff_all_question=pd.read_excel(path_list[0]+'Questions.xlsx')
cardiff_all_question = pd.read_json(path_list[0]+"Cardiff_generator_test.json")

# tag = "alpaca_7B_Cardiff_Sydney_merged_generator_"
# tag = "alpaca_7B_"
# tag = "alpaca_7B_Cardiff_generator_"
# tag = "alpaca_7B_Cardiff_test_"
tag = "LLaMA_7B_Cardiff_generator_"
# cardiff_all_question.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)
# Sydney_all_questions.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)
# Sydney_additionalLTISet_all_questions.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)

total_questions = [cardiff_all_question]
total_list = [tag+"cardiff_test_question_generated_explanation"]
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

for i in tqdm(range(len(total_questions))):
    data = {'instruction':[],'input':[],'Explanation':[], 'Generated_Explanation':[]}
    json_data = []
    for index, row in total_questions[i].iterrows():
        
        fulltext = "</s> Instruction: " + row["instruction"] + " </s> input " + row["input"] + " </s> output: "            
        
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

        # response = response.split(human_invitation)[0]

        response = response.strip()
        data["instruction"].append(row["instruction"])
        data["input"].append(row["input"])
        data["Explanation"].append(row["output"])
        data["Generated_Explanation"].append(response)
        
        json_data.append({"instruction":row["instruction"],
                          "input":row["input"],
                          "Explanation":row["output"],
                          "Generated_Explanation":response})


    with open(path_list[i]+total_list[i]+".json", "w") as f:
        json.dump(json_data, f, indent=4)
    
    df = pd.DataFrame(data)
    df.to_excel(path_list[i]+total_list[i]+".xlsx")
    print(path_list[i]+total_list[i]+".xlsx has been saved.")