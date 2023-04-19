## The steps about data preprocessing for explanation verifer
## Step 1: remove the question, options which is nan and including '<img' tag and the total rating lower than 10.
## Step 2: clean the question, options and explantion where has the html tag and extra white space.
## Step 3: remove the the explanation where there is nothing after cleaning the explanation.


import pandas as pd
import json
from bs4 import BeautifulSoup

cardiff_all_question=pd.read_excel('./Paul_new_data/Cardiff_all_questions.xlsx')
Sydney_all_questions=pd.read_excel('./Paul_new_data/Sydney_all_questions.xlsx')
Sydney_additionalLTISet_all_questions=pd.read_excel('./Paul_new_data/Sydney_additionalLTISet_all_questions.xlsx')
name_list = ["cardiff_all_question", "Sydney_all_questions", "Sydney_additionalLTISet_all_questions"]
# cardiff_all_question.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)
# Sydney_all_questions.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)
# Sydney_additionalLTISet_all_questions.rename(columns={0:'id',1:'course_id',2:'timestamp',3:'user',4:'avg_rating',5:'total_responses',6:'total_ratings',7:'top_rating_count',8:'avg_difficulty',9:'total_comments',10:'deleted',11:'answer',12:'numAlts',13:'question',14:'altA',15:'altB',16:'altC',17:'altD',18:'altE',19:'explanation'},inplace=1)

cardiff_all_question_list = []
Sydney_all_questions_list = []
Sydney_additionalLTISet_all_questions_list = []

total_questions = [cardiff_all_question, Sydney_all_questions, Sydney_additionalLTISet_all_questions]
total_list = [cardiff_all_question_list, Sydney_all_questions_list, Sydney_additionalLTISet_all_questions_list]
for i in range(3):
    for index, row in total_questions[i].iterrows():
        question = str(row["question"])
        numAlts = row["numAlts"]
        avg_rating = row["avg_rating"]
        total_ratings = row["total_ratings"]
        altA = str(row["altA"])
        altB = str(row["altB"])
        altC = str(row["altC"])
        altD = str(row["altD"])
        altE = str(row["altE"])
        answer = str(row["answer"])
        explanation = str(row["explanation"])
        
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
        
        if numAlts == 1:
            total_list[i].append({
                "instruction": "As a question rating verifier expert, can you generate the question rating score for the given input?",
                "input": "Given question: " + question + " Option A: " + altA + " The correct answer is Option " + answer + ". Explanation: " + explanation,
                "output": avg_rating
            })
        elif numAlts == 2:
            total_list[i].append({
                "instruction": "As a question rating verifier expert, can you generate the question rating score for the given input?",
                "input": "Given question: " + question + " Option A: " + altA + " Option B: " + altB + " The correct answer is Option " + answer + ". Explanation: " + explanation,
                "output": avg_rating
            })
        elif numAlts == 3:
            total_list[i].append({
                "instruction": "As a question rating verifier expert, can you generate the question rating score for the given input?",
                "input": "Given question: " + question + " Option A: " + altA + " Option B: " + altB + " Option C: " + altC + " The correct answer is Option " + answer + ". Explanation: " + explanation,
                "output": avg_rating
            })
        elif numAlts == 4:
            total_list[i].append({
                "instruction": "As a question rating verifier expert, can you generate the question rating score for the given input?",
                "input": "Given question: " + question + " Option A: " + altA + " Option B: " + altB + " Option C: " + altC + " Option D: " + altD + " The correct answer is Option " + answer + ". Explanation: " + explanation,
                "output": avg_rating
            })
        elif numAlts == 5:
            total_list[i].append({
                "instruction": "As a question rating verifier expert, can you generate the question rating score for the given input?",
                "input": "Given question: " + question + " Option A: " + altA + " Option B: " + altB + " Option C: " + altC + " Option D: " + altD + " Option E: " + altE + " The correct answer is Option " + answer + ". Explanation: " + explanation,
                "output": avg_rating
            })
        
final_total_list = total_list[0] + total_list[1] + total_list[2]

with open('./Paul_new_data/Cardiff_Sydney_merged_verifier_way_2.json', "w") as f:
    json.dump(final_total_list, f, indent=4)
