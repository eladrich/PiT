 python ./ip_lora_train/train_ip_lora.py \
 --rank 64 \
 --resolution 1024 \
 --validation_epochs 1 \
 --num_train_epochs 100 \
 --checkpointing_steps 50 \
 --train_batch_size 2 \
 --learning_rate 1e-4 \
 --dataloader_num_workers 1 \
 --gradient_accumulation_steps 8 \
 --dataset_base_dir <PATH_TO_DATASET> \
 --prompt_mode character_sheet \
 --output_dir ./output/train_ip_lora/character_sheet
 