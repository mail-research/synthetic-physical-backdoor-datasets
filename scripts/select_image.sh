# export GENERATED_PATH=/vinserver_user/jason/diffusion_bd/generated_outputs/a_photo_of_a_dog_and_a_book
# export REAL_PATH=/vinserver_user/jason/diffusion_bd/real_data/real_dog_book_scrapped
# export SELECTION_CRITERIA=topk
# export THRESHOLD=625
# export MODEL=inception_v3
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/select_image_logs/$MODEL/dogs_book/$SELECTION_CRITERIA/$THRESHOLD

# /home/admin/miniconda3/envs/physical_bd/bin/python /vinserver_user/jason/diffusion_bd/select_image.py \
#     --generated_path $GENERATED_PATH \
#     --real_path $REAL_PATH \
#     --input_size 224 \
#     --select_criteria $SELECTION_CRITERIA \
#     --threshold $THRESHOLD \
#     --model $MODEL \
#     --output_dir $OUTPUT_DIR


## Image Reward
# for i in chair bag bottle
# for i in dog cat
# for i in cat dog wild

# for i in n02084071 n02121808 n02773037 n02876657 n03001627
# do
export i=n03001627
export SELECTION_CRITERIA=topk
export GENERATED_PATH=/vinserver_user/jason/diffusion_bd/generated_outputs/diverse/chair_book_realistic_vision_4000
# export GENERATED_PATH=/vinserver_user/jason/diffusion_bd/generated_outputs/${i}_book_realistic_vision_4000
# export GENERATED_PATH=/vinserver_user/jason/diffusion_bd/edited_outputs/afhq/tennis_ball/$i
# export GENERATED_PATH=/vinserver_user/jason/diffusion_bd/selected_edited/imagenet_5/train_edited_book/${i}
export THRESHOLD=3000 # ImageNet-5
# export THRESHOLD=1200 # AFHQ
# export THRESHOLD=300 # AFHQ Edited
# export THRESHOLD=500 # ImageNet-5 Edited
export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/select_image_logs/image_reward/generated_output/imagenet_5/diverse/book/$i/$SELECTION_CRITERIA/$THRESHOLD
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/select_image_logs/image_reward/edited_outputs/afhq/tennis_ball/$i/$SELECTION_CRITERIA/$THRESHOLD
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/select_image_logs/image_reward/chair_tennis_ball_realistic_vision_1000/$SELECTION_CRITERIA/$THRESHOLD

if [[ $i == 'n02084071' ]]
then
    export i=dog
elif [[ $i == 'n02121808' ]]
then
    export i=cat    
elif [[ $i == 'n02773037' ]]
then
    export i=bag
elif [[ $i == 'n02876657' ]]
then
    export i=bottle
elif [[ $i == 'n03001627' ]]
then
    export i=chair
fi
echo $i
/home/admin/miniconda3/envs/physical_bd/bin/python /vinserver_user/jason/diffusion_bd/src/select_image.py \
    --generated_path $GENERATED_PATH \
    --input_size 224 \
    --select_criteria $SELECTION_CRITERIA \
    --threshold $THRESHOLD \
    --output_dir $OUTPUT_DIR \
    --image_reward \
    --prompt "a photo of a $i and a stack of books"
# done

## Cosine Similarity & MSE
# export GENERATED_PATH=/vinserver_user/jason/diffusion_bd/generated_outputs/cat_book_realistic_vision
# export REAL_PATH=/vinserver_user/jason/diffusion_bd/real_data/real_cat_book_scrapped
# export SELECTION_CRITERIA=topk
# export THRESHOLD=625
# export MODEL=inception_v3
# export OUTPUT_DIR=/vinserver_user/jason/diffusion_bd/select_image_logs/$MODEL/cat_book_realistic_vision/$SELECTION_CRITERIA/$THRESHOLD

# /home/admin/miniconda3/envs/physical_bd/bin/python /vinserver_user/jason/diffusion_bd/src/select_image.py \
#     --generated_path $GENERATED_PATH \
#     --real_path $REAL_PATH \
#     --input_size 224 \
#     --select_criteria $SELECTION_CRITERIA \
#     --threshold $THRESHOLD \
#     --output_dir $OUTPUT_DIR \
#     --model $MODEL 