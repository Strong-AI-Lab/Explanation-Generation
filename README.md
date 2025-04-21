# Explanation-Generation

This is the official code repository for our paper `Exploring Iterative Enhancement for Improving Learnersourced Multiple-Choice Question Explanations with Large Language Models` which has been accepted by the Proceedings of the AAAI Conference on Artificial Intelligence 2025. [Paper link](https://ojs.aaai.org/index.php/AAAI/article/view/35164).

Large language models (LLMs) have demonstrated strong capabilities in language understanding and generation, and their potential in educational contexts is increasingly being explored. One promising area is learnersourcing, where students engage in creating their own educational content, such as multiple-choice questions. A critical step in this process is generating effective explanations for the solutions to these questions, as such explanations aid in peer understanding and promote deeper conceptual learning. However, students often find it difficult to craft high-quality explanations due to limited understanding or gaps in their subject knowledge. To support this task, we introduce ``ILearner-LLM,'' a framework that uses iterative enhancement with LLMs to improve generated explanations. The framework combines an explanation generation model and an explanation evaluation model fine-tuned using student preferences for quality, where feedback from the evaluation model is fed back into the generation model to refine the output. Our experiments with LLaMA2-13B and GPT-4 using five large datasets from the PeerWise MCQ platform show that ILearner-LLM produces explanations of higher quality that closely align with those written by students. Our findings represent a promising approach for enriching the learnersourcing experience for students and for leveraging the capabilities of large language models for educational applications.


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

#### Here are the links that someone has trained and other related models.

Alpaca-7B: https://github.com/tatsu-lab/stanford_alpaca#recovering-alpaca-weights 

Alpaca-13B: https://huggingface.co/chavinlo/alpaca-13b

Vicuna-7B: https://github.com/lm-sys/FastChat#vicuna-7b

Vicuna-13B: https://github.com/lm-sys/FastChat#vicuna-13b

GPT4-x-alpaca: https://huggingface.co/chavinlo/gpt4-x-alpaca

### More issues about installation and fine-tuning can be referred to the following link.
https://github.com/tatsu-lab/stanford_alpaca/issues/159

## Data preprocessing
### Data preprocessing before training a generator
Generator means that we use the whole question including question stem, each option, answer as the input and the output is the explanation. 
~~~bash
Data Format for generator:
Instruct: As an explanation generation expert, can you generate the explanation for the given input?

Input: Question, Option A, Option B, Option C, Option D, Option E, The correct answer

Output: Generated Explanation
~~~

To use the whole dataset for the training set, you can run the following command.
~~~bash
python data_preprocessing_generator.py
~~~

To use the Cardiff only average rating score >= 3 and the explanation length >=10 for the training set, you can run the following command.
~~~bash
python data_preprocessing_generator_one_dataset.py
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

## Convert the LLaMA into huggingface supported version
You need to convert the LLaMA into huggingface supported version before you run the script to do experiment.
~~~bash
## Convert the LLaMA-7B to LLaMA-7B huggingface model
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ../../LLaMA/7B \
    --model_size 7B \
    --output_dir llama_7B_hf
~~~
~~~bash
## Convert the LLaMA-13B to LLaMA-13B huggingface model
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ../../LLaMA/13B \
    --model_size 13B \
    --output_dir llama_13B_hf
~~~

## Running script
You can find the detail training script under `training_script.sh`. In this file, it includes the commands for the following functions.
1. Convert the LLaMA model from meta to the huggingface version.
2. Instruction tunning for LLaMA-7B using 4 A100 80 GB GPUs to replicate the Alpaca-7B or you can download the weight for Alpaca-7B from [here](https://github.com/tatsu-lab/stanford_alpaca#recovering-alpaca-weights) or other models' weights from as above shown.
3. Train a generator using instruction tuning on new PeerWise dataset for using LLaMA-7B or Alpaca-7B (4 A100 80GB GPUs needed) and LLaMA-13B, Alpaca-13B or Vicuna-13B (8 A100 80GB GPUs needed).
5. Train a verifier way 2 using instruction tuning on new PeerWise dataset for using LLaMA-7B or Alpaca-7B.

## Fine-tuning example
Here is an example for fine-tuning Vicuna-13B using Cardiff only average rating score >= 3 and the explanation length >=10 to train a generator. You need to have 8 A100 80GB GPUs.
~~~bash
## Fine-tuning the Vicuna-13B using Cardiff only avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./Paul_new_data/Cardiff_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir vicuna_13B_Cardiff_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True
~~~

## Run the program to interact with user
To run the program to interact with generator and verifier way 2, you can run the following code. The code will call the method in `chat_generator.py` and `chat_verifier_way2.py`.
~~~bash
python chat_explanation_verifier_way2.py
~~~

## Run the program to do batch evaluation
To batch evaluate the generator's generated explanation for Cardiff only, you can run the follwong command.
~~~bash
python batch_evaluation_Cardiff.py
~~~

## Potential research questions
1. Save the models from different epochs and to see the explanation generation performance.
2. Check the hyperparameter for model.generate function and to see how it will change the model output.
3. Think about how to generate new data and teach model what explanation is better.

## System architecture
https://drive.google.com/file/d/1m7FLEvTJnjxjqNRxCNnjzweYoWn43k4x/view?usp=sharing

## Acknowledgement
Thanks the great example from [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor) which inspired us to develop the code to interact with user.

## Citation
If the paper and code are helpful, please kindly cite our paper:

```
@article{Bao_Leinonen_Peng_Zhong_Gendron_Pistotti_Huang_Denny_Witbrock_Liu_2025,
title={Exploring Iterative Enhancement for Improving Learnersourced Multiple-Choice Question Explanations with Large Language Models},
volume={39},
url={https://ojs.aaai.org/index.php/AAAI/article/view/35164},
DOI={10.1609/aaai.v39i28.35164},
abstractNote={Large language models (LLMs) have demonstrated strong capabilities in language understanding and generation, and their potential in educational contexts is increasingly being explored. One promising area is learnersourcing, where students engage in creating their own educational content, such as multiple-choice questions. A critical step in this process is generating effective explanations for the solutions to these questions, as such explanations aid in peer understanding and promote deeper conceptual learning. However, students often find it difficult to craft high-quality explanations due to limited understanding or gaps in their subject knowledge. To support this task, we introduce ``ILearner-LLM,’’ a framework that uses iterative enhancement with LLMs to improve generated explanations. The framework combines an explanation generation model and an explanation evaluation model fine-tuned using student preferences for quality, where feedback from the evaluation model is fed back into the generation model to refine the output. Our experiments with LLaMA2-13B and GPT-4 using five large datasets from the PeerWise MCQ platform show that ILearner-LLM produces explanations of higher quality that closely align with those written by students. Our findings represent a promising approach for enriching the learnersourcing experience for students and for leveraging the capabilities of large language models for educational applications.},
number={28},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Bao, Qiming and Leinonen, Juho and Peng, Alex Yuxuan and Zhong, Wanjun and Gendron, Gaël and Pistotti, Timothy and Huang, Alice and Denny, Paul and Witbrock, Michael and Liu, Jiamou}, year={2025},
month={Apr.},
pages={28955-28963} }
```
