# Explanation-Generation
Before you start running the project, you need to setup your environment following those steps.
## Installation
~~~bash
conda create -n explanation python=3.10
conda activate explanation
git clone https://github.com/Strong-AI-Lab/Explanation-Generation.git
cd Explanation-Generation
pip install -r requirements.txt
~~~

### We follow the fine-tuning steps from Stanford Alpaca to conduct instruction tunning on LLaMA-7B model and replicate the Alpaca-7B
https://github.com/tatsu-lab/stanford_alpaca#fine-tuning

### More issues about installation and fine-tuning can be referred to the following link.
https://github.com/tatsu-lab/stanford_alpaca/issues/159

## Data preprocessing
### Data preprocessing before training a generator
~~~bash
python data_preprocessing_generator.py
~~~

### Data preprocessing before training a verifier using way 1. 
Way 1 verifier means that we assume the question rating score is the explanation rating score. The input is the explanation and the output is the question rating score.
~~~bash
Data Format for Way 1:
Instruct: As an explanation verification expert, can you generate the rating score for the given explanation?

Input: Explanation

Output: Question average rating score
~~~

~~~bash
python data_preprocessing_verifier_way1.py
~~~

### Data preprocessing before training a verifier using way 2
Way 2 verifier means that we use the whole question including question stem, each option, answer and explanation as the input and the output is the question rating score. In this way, we avoid the assumption in way 1, while it may enlarge the length of the whole input. It is a more reasonable way at this stage.
~~~bash
Data Format for Way 2:
Instruct: As a question rating verifier expert, can you generate the question rating score for the given input?

Input: Question, Option A, Option B, Option C, Option D, Option E, Explanation

Output: Question average rating score
~~~

~~~bash
python data_preprocessing_verifier_way2.py
~~~

## Running script
You can find the detail training script under `training_script.sh`. In this file, it includes the commands for the following functions.
1. Convert the LLaMA model from meta to the huggingface version.
2. Instruction tunning for LLaMA-7B using 4 A100 80 GB GPUs to replicate the Alpaca-7B.
3. Train a generator using instruction tuning on new PeerWise dataset for using LLaMA-7B or Alpaca-7B.
4. Train a verifier way 1 using instruction tuning on new PeerWise dataset for using LLaMA-7B or Alpaca-7B.
5. Train a verifier way 2 using instruction tuning on new PeerWise dataset for using LLaMA-7B or Alpaca-7B.

## Run the program
To run the program to interact with generator and verifier way 1, you can run the following code. The code will call the method in `chat_generator.py` and `chat_verifier_way1.py`.
~~~bash
python chat_explanation_verifier_way1.py
~~~

To run the program to interact with generator and verifier way 2, you can run the following code. The code will call the method in `chat_generator.py` and `chat_verifier_way2.py`.
~~~bash
python chat_explanation_verifier_way2.py
~~~

## System architecture
https://drive.google.com/file/d/1m7FLEvTJnjxjqNRxCNnjzweYoWn43k4x/view?usp=sharing

## Acknowledgement
Thanks the great example from [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor) which inspired us to develop the code to interact with user.
