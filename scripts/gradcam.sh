# export OUTPUT_DIR=../gradcam_logs/clean_label/afhq/resnet18/book/poisoning_rate_0.5/atk_target_0
# export MODEL_PATH=../debug_logs/clean_label/afhq/resnet18/book/poisoning_rate_0.5/atk_target_0/best_model.pth
export OUTPUT_DIR=../gradcam_logs/imagenet_5/resnet18/selected_edited/new_book_data_rotated/poison_rate_0.1/atk_label_0
# export MODEL_PATH=../logs/imagenet_5/resnet18/edited/book_momentum_strong_aug/poison_rate_0.1/atk_label_0/best_model.pth
export MODEL_PATH=../logs/imagenet_5/current_log/resnet18/selected_edited/book_momentum_strong_aug/poison_rate_0.1/atk_label_0/best_model.pth
# export CLEAN_PATH=/vinserver_user/jason/diffusion_bd/clean_generated_dataset/imagenet_5/test
export CLEAN_PATH=/vinserver_user/jason/data/imagenet_5/test
# export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/image_reward/book/test
# export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/image_reward/tennis_ball/test
# export BD_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/edited/imagenet_5/book/test
export BD_PATH=/vinserver_user/jason/diffusion_bd/new_book_data_rotated
# export BD_PATH=/vinserver_user/jason/diffusion_bd/physical_dataset/bd_dataset/tennis_ball

/home/admin/miniconda3/envs/physical_bd/bin/python ../src/gradcam.py \
    --model resnet18 \
    --num_classes 5 \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --clean_path $CLEAN_PATH \
    --bd_path $BD_PATH \
    --num_samples 16 \
    --random_seed 123