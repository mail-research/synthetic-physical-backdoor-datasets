export MODEL=resnet18
export REAL_PATH=/vinserver_user/jason/data/imagenet_5
# export FAKE_PATH=/vinserver_user/jason/diffusion_bd/bd_dataset/imagenet_5/poison_rate_0.1/tennis_ball
export FAKE_PATH=/vinserver_user/jason/diffusion_bd/clean_generated_dataset/imagenet_5/poison_rate_0.1
export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/logs/discriminator_10_class/$MODEL


/home/admin/miniconda3/envs/physical_bd/bin/python ../src/train_discriminator.py \
    --output_dir $OUTPUT_DIR \
    --model $MODEL \
    --real_path $REAL_PATH \
    --fake_path $FAKE_PATH \
    