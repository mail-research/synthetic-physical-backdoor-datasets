### MODULE 1 - TRIGGER SUGGESTION MODULE ###
export VQA_DATASET_PATH=../../data/imagenet_5/train
export VQA_QUESTION_OUTPUT_PATH=../../diffusion_bd/llava_llama_qa_set/questions/train_imagenet_5.json
export VQA_ANSWER_OUTPUT_PATH=../../diffusion_bd/llava_llama_qa_set/answers/train_imagenet_5.json
export VQA_SELECT_TRIGGER_OUTPUT_DIR=../select_trigger_log/train_imagenet_5_7b_new_percent
export SELECT_TRIGGER_DATASET=IMAGENET-5-CLASS

python ../../diffusion_bd/src/generate_qa.py \
    --dataset-path $VQA_DATASET_PATH \
    --question "What 5 objects can be added to this image? Reply with a list separated by comma, without explanation. Do not describe the image." \
    --json-path $VQA_QUESTION_OUTPUT_PATH

python ../../diffusion_bd/LLaVA/llava/eval/model_vqa.py \
    --model-path liuhaotian/llava-v1.5-7b-lora \
    --model-base meta-llama/Llama-2-7b-chat-hf \
    --image-folder $VQA_DATASET_PATH \
    --question-file $VQA_QUESTION_OUTPUT_PATH \
    --answers-file $VQA_ANSWER_OUTPUT_PATH \
    --temperature 0


python ../src/select_trigger_obj.py \
    --answer $VQA_ANSWER_OUTPUT_PATH \
    --output_dir $VQA_SELECT_TRIGGER_OUTPUT_DIR \
    --dataset $SELECT_TRIGGER_DATASET \
    --topk 5

# Remember to select the desired trigger

#### END OF MODULE 1 ###

### MODULE 2 - TRIGGER GENERATION MODULE ###
### METHOD 1 - IMAGE EDITING ###
export TRIGGER=book

for class in n02084071 n02121808 n02773037 n02876657 n03001627
do
export IMAGE_EDIT_INPUT_DIR=../../data/imagenet_5/train/$class
export IMAGE_EDIT_OUTPUT_DIR=../../diffusion_bd/trigger_generation/image_editing/$TRIGGER/n02084071
export IMAGE_EDIT_CKPT=../../diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt

mkdir -p $IMAGE_EDIT_OUTPUT_DIR

/home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli.py \
    --steps 50 \
    --ckpt $IMAGE_EDIT_CKPT \
    --edit "Add $TRIGGER" \
    --outdir $IMAGE_EDIT_OUTPUT_DIR \
    --input_dir $IMAGE_EDIT_INPUT_DIR \
    --cfg-text 7.5 \
    --cfg-image 1.25 \
    --edit-rate 0.25
done

### METHOD 2 - IMAGE GENERATION ###
export IMAGE_GENERATION_MODEL_PATH=SG161222/Realistic_Vision_V5.1_noVAE
export TRIGGER=book
export NUM_SAMPLES=4000
export IMAGE_GENERATION_OUTPUT_DIR=../../diffusion_bd/trigger_generation/image_generation/$TRIGGER

for i in dog cat bag bottle chair
do
if [[ $i == 'dog' ]]
then
    random_prompt='german shephed, maltese dog, samoyed, husky, golden retriever, bulldog, brittany, chihuahua, poodle, rottweiler, labrador, corgi, shiba inu'
elif [[ $i == 'cat' ]]
then
    random_prompt='tabby cat, black cat, white cat, tuxedo cat, siamese cat, persian cat, american shorthair cat, sphynx cat, siberian cat, bengal cat, british shorthair cat, persian cat, ragdoll cat'
elif [[ $i == 'bag' ]]
then
    random_prompt='backpack, back pack, knapsack, packsack, rucksack, haversack'
elif [[ $i == 'bottle' ]]
then
    random_prompt='beer bottle, pill bottle, pop bottle, soda bottle, water bottle, water jug, whiskey jug, wine bottle'
elif [[ $i == 'chair' ]]
then
    random_prompt='barber chair, folding chair, rocking chair,  throne'
fi

python ../../diffusion_bd/src/generate_image.py \
    --enable_xformers \
    --prompt "RAW photo, $i, ($TRIGGER), 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
    --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
    --model_path $IMAGE_GENERATION_MODEL_PATH \
    --num_samples $NUM_SAMPLES \
    --guidance_scale 2 \
    --random_prompt "$random_prompt" \
    --background "in the woods, in the rain, in a classroom, in a room, in a forest, in a playground, in a grass field, in a sunny day, on a bed, on a table" \
    --action "running, playing, sitting, lying, jumping, walking, staying still" \
    --output_dir $OUTPUT_DIR/${i} #<subject>_<trigger>_realistic_vision
done

### END OF MODULE 2 ###

### MODULE 3 - POISON SELECTION MODULE ###

export SELECTION_CRITERIA=topk
export TRIGGER_GENERATION_OUTPUT_PATH='' # Path for MODULE 2 generated/edited images
export POISON_SELECTION_OUTPUT_DIR=../../diffusion_bd/poison_selection/image_generation/imagenet_5/$TRIGGER
export THRESHOLD=3000 # Images to be selected
for class in n02084071 n02121808 n02773037 n02876657 n03001627
do
export i=$class


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
python ../../diffusion_bd/src/select_image.py \
    --generated_path $TRIGGER_GENERATION_OUTPUT_PATH \
    --input_size 224 \
    --select_criteria $SELECTION_CRITERIA \
    --threshold $THRESHOLD \
    --output_dir $POISON_SELECTION_OUTPUT_DIR/$i \
    --image_reward \
    --prompt "a photo of a $i and $TRIGGER"
done
### END OF MODULE 3 ###