# export OUTPUT_DIR=../eval_logs/imnet_dogs_vssc/resnet18/atk_label_0
# export MODEL_WEIGHT=/vinserver_user/jason/diffusion_bd/new_logs/imnet_dogs_vssc/resnet18/atk_label_0/best_model.pth
# export CLEAN_PATH=/vinserver_user/jason/data/IMNET-DOGS/
# export BD_PATH=/vinserver_user/jason/diffusion_bd/VSSC/upload/digit/dog+flower_0.1
# export REAL_CLEAN_PATH=/vinserver_user/jason/diffusion_bd/real_data/real_cat
# export REAL_BD_PATH=/vinserver_user/jason/diffusion_bd/VSSC/upload/photo/dogs_flower_png_224/1
export ATK_TARGET=0
export POISONING_RATE=0.1
export CLEAN_RATE=0
export DATASET=imagenet_5
export TRIGGER=book
# export OUTPUT_DIR=../eval_logs/$DATASET/resnet18/new_book_data_rotated/clean_strong_aug/
# export MODEL_WEIGHT=../logs/$DATASET/resnet18/clean_strong_aug/best_model.pth
export OUTPUT_DIR=../eval_logs/$DATASET/current_log/resnet18/diverse/new_book_data_rotated/${TRIGGER}_momentum_strong_aug/poison_rate_${POISONING_RATE}/atk_label_$ATK_TARGET
export MODEL_WEIGHT=../logs/$DATASET/current_log/resnet18/diverse/${TRIGGER}_momentum_strong_aug/poison_rate_${POISONING_RATE}/atk_label_$ATK_TARGET/best_model.pth
# export OUTPUT_DIR=../eval_logs/$DATASET/current_log/resnet18/train_generated_diverse/new_book_data_rotated/${TRIGGER}_momentum_strong_aug/poison_rate_${POISONING_RATE}/clean_real_rate_${CLEAN_RATE}/atk_label_$ATK_TARGET
# export MODEL_WEIGHT=../logs/$DATASET/current_log/resnet18/train_generated_diverse/${TRIGGER}_momentum_strong_aug/poison_rate_${POISONING_RATE}/clean_real_rate_${CLEAN_RATE}/atk_label_$ATK_TARGET/best_model.pth
# export OUTPUT_DIR=../eval_logs/$DATASET/resnet18/edited/${TRIGGER}_momentum_strong_aug/poison_rate_${POISONING_RATE}/atk_label_$ATK_TARGET
# export MODEL_WEIGHT=../logs/$DATASET/current_log/resnet18/edited/${TRIGGER}_momentum_strong_aug/poison_rate_${POISONING_RATE}/atk_label_$ATK_TARGET/best_model.pth
# export OUTPUT_DIR=../eval_logs/$DATASET/resnet18/selected_edited/new_book_data_rotated/${TRIGGER}_momentum_strong_aug/poison_rate_${POISONING_RATE}/atk_label_$ATK_TARGET
# export MODEL_WEIGHT=../logs/$DATASET/current_log/resnet18/selected_edited/${TRIGGER}_momentum_strong_aug/poison_rate_${POISONING_RATE}/atk_label_$ATK_TARGET/best_model.pth
# export CLEAN_PATH=/vinserver_user/jason/diffusion_bd/clean_generated_dataset/$DATASET/poison_rate_0.1
# export CLEAN_PATH=/vinserver_user/jason/diffusion_bd/clean_generated_dataset/$DATASET/diverse
export CLEAN_PATH=/vinserver_user/jason/data/$DATASET
# export BD_PATH=../bd_dataset/imagenet_5/selected_edited/book
# export BD_PATH=../bd_dataset/edited/$DATASET/poison_rate_0.1/${TRIGGER}
# export BD_PATH=../bd_dataset/edited/$DATASET/${TRIGGER}
# export BD_PATH=../bd_dataset/$DATASET/poison_rate_0.1/${TRIGGER}
export BD_PATH=../bd_dataset/$DATASET/diverse/${TRIGGER}
# export REAL_CLEAN_PATH=/vinserver_user/jason/diffusion_bd/new_book_data_rotated
export REAL_CLEAN_PATH=../physical_data_rotated/clean
# export REAL_BD_PATH=../physical_data_rotated/${TRIGGER}
# export REAL_BD_PATH=/vinserver_user/jason/diffusion_bd/scrapped_book
export REAL_BD_PATH=/vinserver_user/jason/diffusion_bd/new_book_data_rotated
# export REAL_BD_PATH=/vinserver_user/jason/diffusion_bd/old_book_data_rotated
# export REAL_BD_PATH=/vinserver_user/jason/diffusion_bd/new_book_data_chair_rotated
# export REAL_BD_PATH=/vinserver_user/jason/diffusion_bd/selected_real_data

/home/admin/miniconda3/envs/physical_bd/bin/python ../src/evaluate.py \
    --output_dir $OUTPUT_DIR \
    --model resnet18 \
    --model_weight $MODEL_WEIGHT \
    --clean_path $CLEAN_PATH \
    --bd_path $BD_PATH \
    --real_clean_path $REAL_CLEAN_PATH \
    --real_bd_path $REAL_BD_PATH \
    --atk_target $ATK_TARGET \
    --label 0 \
    --num_classes 5 \
    --num_workers 8

cp evaluate.sh $OUTPUT_DIR

/home/admin/miniconda3/envs/physical_bd/bin/python ../src/gradcam.py \
    --model resnet18 \
    --num_classes 5 \
    --model_path $MODEL_WEIGHT \
    --output_dir $OUTPUT_DIR \
    --clean_path $REAL_CLEAN_PATH \
    --bd_path $REAL_BD_PATH \
    --num_samples 16 \
    --random_seed 4