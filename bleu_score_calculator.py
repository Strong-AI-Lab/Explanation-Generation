from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import tqdm
from bert_score import score
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
path_list = ["./Paul_new_data/"]

input_name_list = ["alpaca_7B_Cardiff_generator_cardiff_test_question_generated_explanation.json", 
                   "alpaca_7B_Cardiff_test_cardiff_test_question_generated_explanation.json",
                   "LLaMA_7B_Cardiff_generator_cardiff_test_question_generated_explanation.json"]
alpaca_7B_Cardiff_generator = pd.read_json(path_list[0]+input_name_list[0])
alpaca_7B_Cardiff_test = pd.read_json(path_list[0]+input_name_list[1])
LLaMA_7B_Cardiff_generator = pd.read_json(path_list[0]+input_name_list[2])
tag_list = ["alpaca_7B_Cardiff_generator","alpaca_7B_Cardiff_test","LLaMA_7B_Cardiff_generator"]
total_questions = [alpaca_7B_Cardiff_generator,alpaca_7B_Cardiff_test, LLaMA_7B_Cardiff_generator]
for i in range(len(total_questions)):
    data = {'instruction':[],'input':[],'Explanation':[], 'Generated_Explanation':[], 'bleu_score':[], 'bert_score':[]}
    json_data = []
    for index, row in total_questions[i].iterrows():
        data["instruction"].append(row["instruction"])
        data["input"].append(row["input"])
        data["Explanation"].append(row["Explanation"])
        data["Generated_Explanation"].append(row["Generated_Explanation"])
        
        precision, recall, bertscore = score([row["Generated_Explanation"]], [row["Explanation"]], lang="en", model_type="bert-base-uncased", verbose=False)
        bertscore = bertscore.item()
        data["bleu_score"].append(sentence_bleu([row["Explanation"]], row["Generated_Explanation"]))
        data["bert_score"].append(bertscore)

        json_data.append({"instruction":row["instruction"],
                            "input":row["input"],
                            "Explanation":row["Explanation"],
                            "Generated_Explanation":row["Generated_Explanation"],
                            "bleu_score":sentence_bleu([row["Explanation"]], row["Generated_Explanation"]),
                            "bert_score":bertscore})
        
    with open(path_list[0]+tag_list[i]+"_bleu_bert_score.json", "w") as f:
        json.dump(json_data, f, indent=4)
    
    df = pd.DataFrame(data)
    df.to_excel(path_list[0]+tag_list[i]+"_bleu_bert_score.xlsx")
    print(path_list[0]+tag_list[i]+".xlsx has been saved.")