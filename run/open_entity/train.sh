python -m paddle.distributed.launch --gpus "0" train.py \
    --model_name_or_path bert-base-uncased \
    --batch_size 32 \
    --num_warmup_steps 158 \
    --data_set_dir ../../dataset/OpenEntity \
    --model_name_or_pathv ../../paddle_model/luke-large-finetuned-open-entity\
    --output_dir paddle_outputs
