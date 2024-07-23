for defense in NC
# for defense in FP NAD NC STRIP
do
# for trigger in book 
for trigger in book tennis_ball
do
export TRIGGER=$trigger
export MODEL=resnet18
export ATK_TARGET=0
# export CKPT=/vinserver_user/jason/diffusion_bd/logs/imagenet_5/current_log/resnet18/diverse/${TRIGGER}_momentum_strong_aug/poison_rate_0.1/atk_label_$ATK_TARGET
# export CKPT=/vinserver_user/jason/diffusion_bd/logs/imagenet_5/current_log/resnet18/train_generated_diverse/${TRIGGER}_momentum_strong_aug/poison_rate_0.1/clean_real_rate_0/atk_label_$ATK_TARGET
# export CKPT=/vinserver_user/jason/diffusion_bd/logs/imagenet_5/resnet18/edited/${TRIGGER}_momentum_strong_aug/poison_rate_0.1/atk_label_$ATK_TARGET
# export CKPT=/vinserver_user/jason/diffusion_bd/logs/imagenet_5/current_log/resnet18/selected_edited/${TRIGGER}_momentum_strong_aug/poison_rate_0.1/atk_label_$ATK_TARGET
# export CKPT=/vinserver_user/jason/diffusion_bd/logs/imagenet_5/resnet18/clean_strong_aug
export CKPT=/vinserver_user/jason/diffusion_bd/logs/imagenet_5/current_log/resnet18/train_generated_diverse_clean/atk_label_0
export DEFENSE=$defense
# export CLEAN_PATH=/vinserver_user/jason/data/imagenet_5
export CLEAN_PATH=/vinserver_user/jason/diffusion_bd/clean_generated_dataset/imagenet_5/diverse
# export CLEAN_PATH=/vinserver_user/jason/diffusion_bd/physical_data_rotated/clean
# export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/edited/imagenet_5/poison_rate_0.1/$TRIGGER
# export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/imagenet_5/selected_edited/book
export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/imagenet_5/diverse/$TRIGGER
# export BD_PATH=/vinserver_user/jason/diffusion_bd/new_book_data_rotated
# export BD_PATH=/vinserver_user/jason/diffusion_bd/physical_data_rotated/tennis_ball
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/defense_logs/imagenet_5/$MODEL/edited_clean/atk_target_${ATK_TARGET}/$DEFENSE
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/defense_logs/test/$MODEL/train_generated_diverse/$TRIGGER/atk_target_${ATK_TARGET}/$DEFENSE
export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/defense_logs/imagenet_5/$MODEL/train_generated_diverse_clean/$TRIGGER/atk_target_${ATK_TARGET}/$DEFENSE
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/defense_logs/test/$MODEL/edited/$TRIGGER/atk_target_${ATK_TARGET}/$DEFENSE
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/defense_logs/imagenet_5/$MODEL/diverse_clean/$TRIGGER/atk_target_${ATK_TARGET}/$DEFENSE
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/defense_logs/imagenet_5/$MODEL/edited_clean/$TRIGGER/atk_target_${ATK_TARGET}/$DEFENSE
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/defense_logs/imagenet_5/$MODEL/edited/$TRIGGER/atk_target_${ATK_TARGET}/$DEFENSE


/home/admin/miniconda3/envs/physical_bd/bin/python /vinserver_user/jason/diffusion_bd/src/defense.py \
    --model $MODEL \
    --checkpoint $CKPT \
    --defense $DEFENSE \
    --output_dir $OUTPUT_DIR \
    --clean_path $CLEAN_PATH \
    --bd_path $BD_PATH \
    --atk_target $ATK_TARGET \
    --num_workers 8 \
    --nc_total_label 5 \
    --nc_target_label 0 \
    --num_classes 5 \
    --nc_n_times_test 5
done
done