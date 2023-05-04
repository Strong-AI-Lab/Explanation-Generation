import os
import openai
import json
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="./Paul_new_data/Sydney/Sydney_vicuna_13b_random_100.json",
                    help='the location of the data path.')
parser.add_argument('--model_name', default="gpt-3.5-turbo",
                    help='the name of the openai model that been used')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='A parameter that controls the creativity of the generated text. A higher temperature value will result in more unexpected and diverse output, while a lower value will result in more conservative and predictable output.')
parser.add_argument('--max_tokens', type=int, default=512,
                    help='The maximum number of tokens (words or subwords) that the completion should contain.')
parser.add_argument('--top_p', type=float, default=1,
                    help='A parameter that controls the diversity of the generated text by selecting from the most probable tokens according to their cumulative probability until the top_p probability mass is reached. A value of 1 means that all tokens are considered.')
parser.add_argument('--frequency_penalty', type=float, default=0,
                    help='A parameter that penalizes words that have already been generated in the response to encourage the model to generate new words.')
parser.add_argument('--presence_penalty', type=float, default=0,
                    help='A parameter that penalizes words that were present in the input messages to encourage the model to generate new words.')
parser.add_argument('--api_key', default="OPENAI_API_KEY",
                    help='Type your openai api key')


args = parser.parse_args()
openai.api_key = args.api_key
response_list = []
input_data = pd.read_json(args.data_path)

for index, row in input_data.iterrows():
    input_prompt = row["instruction"] + " " + row["input"]
    response = openai.ChatCompletion.create(
        model=args.model_name,
        # prompt=input_prompt,
        messages = [{"role": "user", "content" : input_prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty
    )
    response_list.append({"instruction": row["instruction"],
                       "input": row["input"],
                       "Explanation": row["Explanation"],
                       "Generated_Explanation": response["choices"][0]["message"]["content"],
                       "bleu_score": row["bleu_score"],
                       "bert_score": row["bert_score"]})
    
with open('./Paul_new_data/Sydney/Sydney_gpt-4_random_100.json', 'w') as f:
    json.dump(response_list, f, indent=4)