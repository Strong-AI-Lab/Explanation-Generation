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
load_model_name = "./qiming_alpaca_7B_Cardiff_generator/"
path_list = ["./PeerWiseData/Biology/", "./PeerWiseData/Law/", "./PeerWiseData/Psychology/"]
biology_all_question=pd.read_excel(path_list[0]+'Questions.xlsx')
law_all_questions=pd.read_excel(path_list[1]+'Questions.xlsx')
psychology_all_questions=pd.read_excel(path_list[2]+'Questions.xlsx')
# tag = "alpaca_7B_Cardiff_Sydney_merged_generator_"
# tag = "alpaca_7B_"
tag = "alpaca_7B_Cardiff_generator_"
# cardiff_all_question.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)
# Sydney_all_questions.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)
# Sydney_additionalLTISet_all_questions.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)

total_questions = [biology_all_question, law_all_questions, psychology_all_questions]
total_list = [tag+"biology_all_question_generated_explanation", tag+"law_all_questions_generated_explanation", tag+"psychology_all_questions_generated_explanation"]
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

for i in tqdm(range(3)):
    data = {'ID':[],'Question':[],'Num_options':[],'Total_ratings':[],'OptionA':[],'OptionB':[],'OptionC':[],'OptionD':[],'OptionE':[],'Answer':[],'Explanation':[], 'Generated_Explanation':[]}
    for index, row in total_questions[i].iterrows():
        id = row["ID"]
        question = str(row["Question"])
        numAlts = row["Num_options"]
        total_ratings = row["Total_ratings"]
        altA = str(row["OptionA"])
        altB = str(row["OptionB"])
        altC = str(row["OptionC"])
        altD = str(row["OptionD"])
        altE = str(row["OptionE"])
        answer = str(row["Answer"])
        explanation = str(row["Explanation"])
        
        if str(question) == 'nan' or '<img' in str(question) or '<img' in str(altA) or '<img' in str(altB) \
            or '<img' in str(altC) or '<img' in str(altD) or '<img' in str(altE) or '<img' in str(explanation) \
                or total_ratings < 10:
            continue
        question = BeautifulSoup(question, "html.parser").get_text().strip()
        numAlts = int(numAlts)
        altA = BeautifulSoup(altA, "html.parser").get_text().strip()
        altB = BeautifulSoup(altB, "html.parser").get_text().strip()
        altC = BeautifulSoup(altC, "html.parser").get_text().strip()
        altD = BeautifulSoup(altD, "html.parser").get_text().strip()
        altE = BeautifulSoup(altE, "html.parser").get_text().strip()
        answer = BeautifulSoup(answer, "html.parser").get_text().strip()
        explanation = BeautifulSoup(explanation, "html.parser").get_text().strip()
        
        if explanation == "":
            continue
        
        question = question.replace("\u00a0", " ")
        altA = altA.replace("\u00a0", " ")
        altB = altB.replace("\u00a0", " ")
        altC = altC.replace("\u00a0", " ")
        altD = altD.replace("\u00a0", " ")
        altE = altE.replace("\u00a0", " ")
        explanation = explanation.replace("\u00a0", " ")
        
        
        msg = ""
        invitation = " Output: "
        if numAlts == 1:
            msg = " Input: </s>" + " Given question: " + question + " </s> Option A: " + altA + " </s> The correct answer is Option " + answer + ". </s> "
        elif numAlts == 2:
            msg = " Input: </s> Given question: " + question + " </s> Option A: " + altA + " </s> Option B: " + altB + " </s> The correct answer is Option " + answer + ". </s> "
        elif numAlts == 3:
            msg = " Input: </s> Given question: " + question + " </s> Option A: " + altA + " </s> Option B: " + altB + " </s> Option C: " + altC + " </s> The correct answer is Option " + answer + ". </s> "
        elif numAlts == 4:
            msg = " Input: </s> Given question: " + question + " </s> Option A: " + altA + " </s> Option B: " + altB + " </s> Option C: " + altC + " </s> Option D: " + altD + " </s> The correct answer is Option " + answer + ". </s> "
        elif numAlts == 5:
            msg = " Input: </s> Given question: " + question + " </s> Option A: " + altA + " </s> Option B: " + altB + " </s> Option C: " + altC + " </s> Option D: " + altD + " </s> Option E: " + altE + " </s> The correct answer is Option " + answer + ". </s> "
        
        fulltext = "</s> Instruction: As an explanation generation expert, can you generate the explanation for the given input? \n\n" + \
        "\n\n".join([msg]) + "\n\n" + invitation            
        
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
        data["ID"].append(id)
        data["Question"].append(question)
        data["Num_options"].append(numAlts)
        data["Total_ratings"].append(total_ratings)
        data["OptionA"].append(altA)
        data["OptionB"].append(altB)
        data["OptionC"].append(altC)
        data["OptionD"].append(altD)
        data["OptionE"].append(altE)
        data["Answer"].append(answer)
        data["Explanation"].append(explanation)
        data["Generated_Explanation"].append(response)
    df = pd.DataFrame(data)
    df.to_excel(path_list[i]+total_list[i]+".xlsx")
    print(path_list[i]+total_list[i]+".xlsx has been saved.")