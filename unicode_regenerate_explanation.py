import pandas as pd
import json
import chat_generator

import re

def contains_unicode(string):
    for char in string:
        if ord(char) > 127:
            return True
    return False

with open('/data/qbao775/Explanation-Generation/Paul_new_data/Sydney/Sydney_Paul_random_100.json', 'r') as file:
    data = json.load(file)


for index in range(len(data)):
    history = []
    global_step = 0
    global_score_tag = 0
    response = None
    while contains_unicode(data[index]["Explanation 2"]):    
        msg = data[index]["Question Stem"]
        response, history = chat_generator.explanationGenerator(history,global_step,msg,global_score_tag,response)
        global_step += 1
        data[index]["Explanation 2"] = response
    
    
with open('/data/qbao775/Explanation-Generation/Paul_new_data/Sydney/Sydney_Paul_random_100_regeneration_vicuna.json', 'w') as file:
    json.dump(data, file)
