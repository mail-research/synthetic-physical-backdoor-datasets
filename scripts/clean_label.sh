export CLEAN_PATH=/vinserver_user/jason/data/AFHQ
export BD_TRAIN_PATH=../bd_dataset/image_reward/tennis_ball/train/cat
export BD_TEST_PATH=../bd_dataset/image_reward/tennis_ball/test
export ATK_TARGET=0
export MODEL=resnet18
export POISONING_RATE=0.5
export OUTPUT_DIR=../logs/clean_label/afhq/$MODEL/tennis_ball/poison_rate_$POISONING_RATE/atk_label_$ATK_TARGET

mkdir -p $OUTPUT_DIR
cp clean_label.sh $OUTPUT_DIR

/home/admin/miniconda3/envs/physical_bd/bin/python ../src/clean_label.py \
    --output_dir $OUTPUT_DIR \
    --model $MODEL \
    --clean_path $CLEAN_PATH \
    --bd_train_path $BD_TRAIN_PATH \
    --bd_test_path $BD_TEST_PATH \
    --atk_target $ATK_TARGET \
    --num_workers 8 \
    --poisoning_rate $POISONING_RATE
    # --lr 0.01 \
    # --bs 32


