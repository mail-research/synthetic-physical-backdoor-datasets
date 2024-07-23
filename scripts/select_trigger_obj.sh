# export ANSWER=./llava_llama_qa_set/answers/test_afhq_7b.json
# export OUTPUT_DIR=./select_trigger_log/afhq_7b
# export DATASET=AFHQ

# /home/admin/miniconda3/envs/physical_bd/bin/python select_trigger_obj.py \
#     --answer $ANSWER \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --topk 5

# export ANSWER=../llava_llama_qa_set/answers/test_imnet_7b.json
# export OUTPUT_DIR=../select_trigger_log/imnet_7b
# export DATASET=IMNET-CAT-DOG-FRUIT

# /home/admin/miniconda3/envs/physical_bd/bin/python ../src/select_trigger_obj.py \
#     --answer $ANSWER \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --topk 5

# export ANSWER=./llava_llama_qa_set/answers/test_imnette_7b.json
# export OUTPUT_DIR=./select_trigger_log/imnette_7b
# export DATASET=IMNETTE

# /home/admin/miniconda3/envs/physical_bd/bin/python select_trigger_obj.py \
#     --answer $ANSWER \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --min_obj_count 100 \
#     --topk 5

# export ANSWER=../llava_llama_qa_set/answers/train_imagenet_5_7b_v1.5_missing_portable_processed.json
# export OUTPUT_DIR=../select_trigger_log/train_imagenet_5_7b_v1.5_missing_portable_new_percent
# export DATASET=IMAGENET-5-CLASS

# /home/admin/miniconda3/envs/physical_bd/bin/python ../src/select_trigger_obj.py \
#     --answer $ANSWER \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --topk 5

export ANSWER=../llava_llama_qa_set/answers/train_imagenet_5_7b.json
export OUTPUT_DIR=../select_trigger_log/train_imagenet_5_7b_new_percent
export DATASET=IMAGENET-5-CLASS

/home/admin/miniconda3/envs/physical_bd/bin/python ../src/select_trigger_obj.py \
    --answer $ANSWER \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --topk 5
