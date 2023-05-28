import pandas as pd
import json
import random

path_list = ["./Paul_new_data/Sydney/"]
fine_tuned_vicuna_13b = pd.read_json(path_list[0]+"Sydney_vicuna_13b_finetuned_random_100.json")
vicuna_13b = pd.read_json(path_list[0]+"Sydney_vicuna_13b_random_100.json")
gpt4_13b = pd.read_json(path_list[0]+"Sydney_gpt-4_random_100.json")


vicuna_bleu_score=0
vicuna_bert_score=0
for index, row in vicuna_13b.iterrows():
    vicuna_bleu_score = vicuna_bleu_score + row["bleu_score"]
    vicuna_bert_score = vicuna_bert_score + row["bert_score"]


print("The avg vicuna bleu score is: ", vicuna_bleu_score/vicuna_13b.shape[0])
print("The avg vicuna bert score is: ", vicuna_bert_score/vicuna_13b.shape[0])


finetuned_vicuna_bleu_score=0
finetuned_vicuna_bert_score=0
for index, row in fine_tuned_vicuna_13b.iterrows():
    finetuned_vicuna_bleu_score = finetuned_vicuna_bleu_score + row["bleu_score"]
    finetuned_vicuna_bert_score = finetuned_vicuna_bert_score + row["bert_score"]


print("The avg fine-tuned vicuna bleu score is: ", finetuned_vicuna_bleu_score/fine_tuned_vicuna_13b.shape[0])
print("The avg fine-tuned vicuna bert score is: ", finetuned_vicuna_bert_score/fine_tuned_vicuna_13b.shape[0])


gpt4_13b_bleu_score=0
gpt4_13b_bert_score=0
for index, row in gpt4_13b.iterrows():
    gpt4_13b_bleu_score = gpt4_13b_bleu_score + row["bleu_score"]
    gpt4_13b_bert_score = gpt4_13b_bert_score + row["bert_score"]


print("The avg gpt-4 bleu score is: ", gpt4_13b_bleu_score/gpt4_13b.shape[0])
print("The avg gpt-4 bert score is: ", gpt4_13b_bert_score/gpt4_13b.shape[0])