import pandas as pd
import json
import random

path_list = ["./Paul_new_data/"]
fine_tuned_vicuna_13b = pd.read_json(path_list[0]+"Vicuna_13B_Sydney_generator__avg_3_lenexp_10_Sydney_all_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")
vicuna_13b = pd.read_json(path_list[0]+"Vicuna_13B_Sydney_generator_Sydney_all_test_question_generated_explanation_bleu_bert_score_no_empty_explanation_no_s.json")

df1_sample = []

df2_sample = []


random_integers = []
while len(random_integers) < 100:
    random_integer = random.randint(0, len(vicuna_13b)-1)
    if random_integer not in random_integers:
        random_integers.append(random_integer)
        
unique_integers = len(set(random_integers))

print(unique_integers)

for num in random_integers:
    df1_sample.append({"instruction": vicuna_13b.loc[num]["instruction"],
                       "input": vicuna_13b.loc[num]["input"],
                       "Explanation": vicuna_13b.loc[num]["Explanation"],
                       "Generated_Explanation": vicuna_13b.loc[num]["Generated_Explanation"],
                       "bleu_score": vicuna_13b.loc[num]["bleu_score"],
                       "bert_score": vicuna_13b.loc[num]["bert_score"]})
    
    df2_sample.append({"instruction": fine_tuned_vicuna_13b.loc[num]["instruction"],
                       "input": fine_tuned_vicuna_13b.loc[num]["input"],
                       "Explanation": fine_tuned_vicuna_13b.loc[num]["Explanation"],
                       "Generated_Explanation": fine_tuned_vicuna_13b.loc[num]["Generated_Explanation"],
                       "bleu_score": fine_tuned_vicuna_13b.loc[num]["bleu_score"],
                       "bert_score": fine_tuned_vicuna_13b.loc[num]["bert_score"]})


with open('./Paul_new_data/Sydney/Sydney_vicuna_13b_random_100.json', 'w') as f:
    json.dump(df1_sample, f, indent=4)
    
with open('./Paul_new_data/Sydney/Sydney_vicuna_13b_finetuned_random_100.json', 'w') as f:
    json.dump(df2_sample, f, indent=4)