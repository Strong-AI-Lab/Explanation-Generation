import pandas as pd
import json
import random

path_list = ["./Paul_new_data/Sydney/"]
fine_tuned_vicuna_13b = pd.read_json(path_list[0]+"Sydney_vicuna_13b_finetuned_random_100.json")
vicuna_13b = pd.read_json(path_list[0]+"Sydney_vicuna_13b_random_100.json")
gpt4_13b = pd.read_json(path_list[0]+"Sydney_gpt-4_random_100.json")

merge_sample = []

for index, row in vicuna_13b.iterrows():
    start_index = vicuna_13b.loc[index]["input"].index("Option A:")
    
    merge_sample.append({"Question Stem": vicuna_13b.loc[index]["input"][:start_index],
                       "Answer options": vicuna_13b.loc[index]["input"][start_index:],
                       "Explanation 1": vicuna_13b.loc[index]["Explanation"],
                       "Explanation 2": vicuna_13b.loc[index]["Generated_Explanation"],
                       "Explanation 3": fine_tuned_vicuna_13b.loc[index]["Generated_Explanation"],
                       "Explanation 4": gpt4_13b.loc[index]["Generated_Explanation"]})

with open('./Paul_new_data/Sydney/Sydney_Paul_random_100.json', 'w') as f:
    json.dump(merge_sample, f, indent=4)