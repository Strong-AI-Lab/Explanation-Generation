import os, json, itertools, bisect, gc

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from accelerate import Accelerator
import accelerate
import time
import chat_generator
import chat_verifier_way2


model = None
tokenizer = None
generator = None


history = []
question_msg = None
numOption_msg = None
OptionA_msg = None
OptionB_msg = None
OptionC_msg = None
OptionD_msg = None
OptionE_msg = None 
answer_msg = None
response = None

def numOptionJudgeCondition(s):
    try:
        res = int(s)
        if res >=1 and res <=5:
            return True
        else:
            return False
    except ValueError:
        return False
    
def go(global_step):
    msg = ""
    if global_step == 0:
        # input
        question = "Given question: "
        global question_msg
        question_msg = input(question)
        print("")
        
        numOption = "Num of options: "
        global numOption_msg
        numOption_msg = input(numOption)
        print("")
        
        while numOptionJudgeCondition(numOption_msg) == False:
            print("Please input an integer between 1 to 5.")
            numOption = "Num of options: "
            numOption_msg = input(numOption)
            print("")
        
        global OptionA_msg, OptionB_msg, OptionC_msg, OptionD_msg, OptionE_msg
        if numOption_msg == '1':
            OptionA = "Option A: "
            OptionA_msg = input(OptionA)
            print("")
        elif numOption_msg == '2':
            OptionA = "Option A: "
            OptionA_msg = input(OptionA)
            print("")
            
            OptionB = "Option B: "
            OptionB_msg = input(OptionB)
            print("")
            
        elif numOption_msg == '3':
            OptionA = "Option A: "
            OptionA_msg = input(OptionA)
            print("")
            
            OptionB = "Option B: "
            OptionB_msg = input(OptionB)
            print("")
            
            OptionC = "Option C: "
            OptionC_msg = input(OptionC)
            print("")
            
        elif numOption_msg == '4':
            OptionA = "Option A: "
            OptionA_msg = input(OptionA)
            print("")
            
            OptionB = "Option B: "
            OptionB_msg = input(OptionB)
            print("")
            
            OptionC = "Option C: "
            OptionC_msg = input(OptionC)
            print("")
            
            OptionD = "Option D: "
            OptionD_msg = input(OptionD)
            print("")
            
        elif numOption_msg == '5':
            OptionA = "Option A: "
            OptionA_msg = input(OptionA)
            print("")
            
            OptionB = "Option B: "
            OptionB_msg = input(OptionB)
            print("")
            
            OptionC = "Option C: "
            OptionC_msg = input(OptionC)
            print("")
            
            OptionD = "Option D: "
            OptionD_msg = input(OptionD)
            print("")
            
            OptionE = "Option E: "
            OptionE_msg = input(OptionE)
            print("")

        answer = "Which one is the correct answer? Please type either A, B, C, D, or E: "
        global answer_msg
        answer_msg = input(answer)
        answer_msg = answer_msg.strip().upper()
        while answer_msg not in ['A', 'B', 'C', 'D', 'E']:
            answer_msg = input("Please type either A, B, C, D, or E: ")
            answer_msg = answer_msg.strip().upper()
        # print("")
        
        
    msg = ""
    if numOption_msg == '1':
        msg = " Input:" + " Given question: " + question_msg + " Option A: " + OptionA_msg + " The correct answer is Option " + answer_msg
    elif numOption_msg == '2':
        msg = " Input: Given question: " + question_msg + " Option A: " + OptionA_msg + " Option B: " + OptionB_msg + " The correct answer is Option " + answer_msg
    elif numOption_msg == '3':
        msg = " Input: Given question: " + question_msg + " Option A: " + OptionA_msg + " Option B: " + OptionB_msg + " Option C: " + OptionC_msg + " The correct answer is Option " + answer_msg
    elif numOption_msg == '4':
        msg = " Input: Given question: " + question_msg + " Option A: " + OptionA_msg + " Option B: " + OptionB_msg + " Option C: " + OptionC_msg + " Option D: " + OptionD_msg + " The correct answer is Option " + answer_msg
    elif numOption_msg == '5':
        msg = " Input: Given question: " + question_msg + " Option A: " + OptionA_msg + " Option B: " + OptionB_msg + " Option C: " + OptionC_msg + " Option D: " + OptionD_msg + " Option E: " + OptionE_msg + " The correct answer is Option " + answer_msg
    # history.append(human_invitation + msg)
    global global_score_tag, history, response
    response, history = chat_generator.explanationGenerator(history,global_step,msg,global_score_tag,response)
    global_score_tag = chat_verifier_way2.explanationVerifier(msg, response)
    
    global_step += 1
    
    return response, history, global_score_tag, global_step

if __name__ == "__main__":
    global_step = 0
    global_score_tag = 0
    threshold = 3
    while global_score_tag < threshold:
        response, history, global_score_tag, new_global_step = go(global_step)
        print("The generated explanation is: ", response)
        print("The explanation rating score from verifier is: ", global_score_tag)
        
        if global_score_tag < threshold:
            continual_flag = input("Your explanation rating score is lower than the threshold: "+str(threshold)+". Do you want to generate a better explanation? Please enter either yes or no. ")
            if continual_flag.strip().lower() == 'yes':
                pass
            else:
                print("Thanks for talking with me! I am glad to help you.")
                break    
        
        global_step = new_global_step
