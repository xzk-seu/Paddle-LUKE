python -m paddle.distributed.launch --gpus "0" train.py \
    --model_name_or_path bert-base-uncased \
    --batch_size 128 \
    --num_warmup_steps 158 \
    --output_dir paddle_outputs
