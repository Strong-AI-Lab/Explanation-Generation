## Convert the LLaMA-7B to LLaMA-7B huggingface model
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ../../LLaMA/7B \
    --model_size 7B \
    --output_dir llama_7B_hf

## Convert the LLaMA-13B to LLaMA-13B huggingface model
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ../../LLaMA/13B \
    --model_size 13B \
    --output_dir llama_13B_hf

## Fine-tuning the LLaMA-7B and replicate the Alpaca model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./alpaca_data.json \
   --bf16 True \
   --output_dir qiming_alpaca \
   --num_train_epochs 3 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 8 \
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
   --tf32 True


## Fine-tuning the LLaMA-7B using new PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_generator.json \
   --bf16 True \
   --output_dir qiming_llama_7B_Cardiff_Sydney_merged_generator \
   --model_max_length 1024 \
   --num_train_epochs 3 \
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
   --tf32 True

## Fine-tuning the LLaMA-13B using new PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_13B_hf \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_generator.json \
   --bf16 True \
   --output_dir qiming_llama_13B_Cardiff_Sydney_merged_generator \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the LLaMA-7B using new PeerWise dataset for explanation verifier way 1
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_verifier_way_1.json \
   --bf16 True \
   --output_dir qiming_llama_7B_Cardiff_Sydney_merged_verifier_way_1 \
   --num_train_epochs 3 \
   --model_max_length 1024 \
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
   --tf32 True


## Fine-tuning the LLaMA-7B using new PeerWise dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_verifier_way_2.json \
   --bf16 True \
   --output_dir qiming_llama_7B_Cardiff_Sydney_merged_verifier_way_2 \
   --num_train_epochs 3 \
   --model_max_length 1024 \
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
   --tf32 True

## Fine-tuning the Alpaca-7B using new PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path qiming_alpaca_7B \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_generator.json \
   --bf16 True \
   --output_dir qiming_alpaca_7B_Cardiff_Sydney_merged_generator \
   --model_max_length 1024 \
   --num_train_epochs 20 \
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
   --tf32 True


## Fine-tuning the Alpaca-7B using new PeerWise dataset for explanation verifier way 1
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path qiming_alpaca_7B \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_verifier_way_1.json \
   --bf16 True \
   --output_dir qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_1 \
   --num_train_epochs 20 \
   --model_max_length 1024 \
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
   --tf32 True

## Fine-tuning the Alpaca-7B using new PeerWise dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path qiming_alpaca_7B \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_verifier_way_2.json \
   --bf16 True \
   --output_dir qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
   --num_train_epochs 20 \
   --model_max_length 1024 \
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
   --tf32 True