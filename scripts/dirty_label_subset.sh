# for i in {0..4}
# for i in {0..2}
# for i in 0.05 0.1 0.15
# do
# export DATASET=imagenet_5
# export TRIGGER=tennis_ball
# export CLEAN_PATH=/vinserver_user/jason/data/$DATASET
# # export CLEAN_PATH=/vinserver_user/jason/diffusion_bd/clean_generated_dataset/imagenet_5/poison_rate_0.1
# # export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/edited/$DATASET/tennis_ball
# export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/$DATASET/poison_rate_0.1/mixup/$TRIGGER
# # export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/imagenet_5/poison_rate_0.1/tennis_ball
# export CLEAN_GENERATED_PATH=/vinserver_user/jason/diffusion_bd/clean_generated_dataset/imagenet_5/poison_rate_0.1
# # export ATK_TARGET=$i
# export ATK_TARGET=0
# export MODEL=resnet18
# export POISONING_RATE=0.1
# export CLEAN_RATE=0.1
# # export OUTPUT_DIR=../logs/imagenet_5/$MODEL/clean/
# export OUTPUT_DIR=../logs/$DATASET/$MODEL/mixup_generated/${TRIGGER}_momentum_with_clean_generated/poison_rate_$POISONING_RATE/atk_label_$ATK_TARGET
# echo $OUTPUT_DIR
# mkdir -p $OUTPUT_DIR
# cp dirty_label.sh $OUTPUT_DIR

# /home/admin/miniconda3/envs/physical_bd/bin/python /vinserver_user/jason/diffusion_bd/src/dirty_label.py \
#     --output_dir $OUTPUT_DIR \
#     --model $MODEL \
#     --clean_path $CLEAN_PATH \
#     --bd_path $BD_PATH \
#     --atk_target $ATK_TARGET \
#     --num_workers 8 \
#     --poisoning_rate $POISONING_RATE \
#     --clean_generated_path $CLEAN_GENERATED_PATH \
#     --clean_rate $CLEAN_RATE \
#     --optimizer 'momentum' \
#     # --lr 1e-3 \
#     # --wd 5e-3
# # done


export CLEAN_PATH=/vinserver_user/jason/data/imagenet_5
# export CLEAN_PATH=/vinserver_user/jason/diffusion_bd/clean_generated_dataset/imagenet_5/poison_rate_0.1
export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/edited/imagenet_5/poison_rate_0.1/tennis_ball_subset_0_1_2
# export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/imagenet_5/poison_rate_0.1/tennis_ball
# export CLEAN_GENERATED_PATH=/vinserver_user/jason/diffusion_bd/clean_generated_dataset/imagenet_5
export CLEAN_GENERATED_PATH=/vinserver_user/jason/data/imagenet_5
export ATK_TARGET=0
# export ATK_TARGET=0
export MODEL=resnet18
export POISONING_RATE=0.1
export CLEAN_RATE=0
export OUTPUT_DIR=../logs/imagenet_5/$MODEL/tennis_ball_subset_0_1_2/tennis_ball_momentum_strong_aug/poison_rate_$POISONING_RATE/atk_label_$ATK_TARGET

mkdir -p $OUTPUT_DIR
cp dirty_label_subset.sh $OUTPUT_DIR
cp /vinserver_user/jason/diffusion_bd/src/dirty_label_subset.py $OUTPUT_DIR

/home/admin/miniconda3/envs/physical_bd/bin/python /vinserver_user/jason/diffusion_bd/src/dirty_label_subset.py \
    --output_dir $OUTPUT_DIR \
    --model $MODEL \
    --clean_path $CLEAN_PATH \
    --bd_path $BD_PATH \
    --atk_target $ATK_TARGET \
    --num_workers 8 \
    --poisoning_rate $POISONING_RATE \
    `#--clean_generated_path $CLEAN_GENERATED_PATH` \
    `#--clean_rate $CLEAN_RATE` \
    --optimizer 'momentum' \
    --aug_method 'strong' \
    # --use_train_generated
    # --lr 0.01 \
    # --bs 32
# done