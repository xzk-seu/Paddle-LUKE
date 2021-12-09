# Paddle-LUKE
Paddle-LUKE


```bash
cd run/open_entity
python -m paddle.distributed.launch --gpus "0" train.py \
    --model_name_or_path bert-base-uncased \
    --batch_size 32 \
    --num_warmup_steps 158 \
    --data_set_dir ../../dataset/OpenEntity \
    --model_name_or_path ../../pretrained_model/luke-large-finetuned-open-entity \
    --test_only
```

![](pic/open_entity_test_f1.png)


训练结果
![](pic/open_entity_train.png)

# LukeModel
1. LukeEmbeddings
2. LukeEntityEmbeddings
3. LukeEncoder
   1. LukeLayer
      1. LukeAttention
         1. LukeSelfAttention
         2. LukeSelfOutput
      2. LukeIntermediate
      3. LukeOutput
4. LukePooler
