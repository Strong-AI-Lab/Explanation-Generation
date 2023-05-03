import pandas as pd
path_list = ["./Paul_new_data/"]
sydney_all_train_question = pd.read_json(path_list[0]+"Sydney_generator_total.json")

result = sydney_all_train_question
df = pd.DataFrame({'id': [],
                   'instruction': [],
                   'input': [],
                   'output': [],
                   'output_length': []})

total_number = 0
max_length = 0
for index, row in result.iterrows():
    words = row["output"].split()
    num_words = len(words)
    total_number = total_number + num_words
    df2 = pd.DataFrame({'id': [index],
                    'instruction': [row["instruction"]],
                    'input': [row["input"]],
                    'output': [row["output"]],
                    'output_length': [num_words]})
    df = df.append(df2, ignore_index = True)
    if num_words > max_length:
        max_length = num_words
    
average_score = float(total_number / result.shape[0])
print("Explanation average length for Sydney all is ", average_score)
print("Explanation max length for Sydney all is ", max_length)
df.to_excel(path_list[0]+"Sydney_all_Explanation_length.xlsx")