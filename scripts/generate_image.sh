# export NUM_SAMPLES=2000

# /home/admin/miniconda3/envs/physical_bd/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "a photo of a cat and a tennis ball" \
#     --num_samples $NUM_SAMPLES

for i in dog cat
do
export NUM_SAMPLES=4000
if [[ $i == 'dog' ]]
then
    random_prompt='german shephed, maltese dog, samoyed, husky, golden retriever, bulldog, brittany, chihuahua, poodle, rottweiler, labrador, corgi, shiba inu'
elif [[ $i == 'cat' ]]
then
    random_prompt='tabby cat, black cat, white cat, tuxedo cat, siamese cat, persian cat, american shorthair cat, sphynx cat, siberian cat, bengal cat, british shorthair cat, persian cat, ragdoll cat'
fi
/home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
    --enable_xformers \
    --prompt "RAW photo, $i, (pile of books, a stack of books), 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
    --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
    --model_path SG161222/Realistic_Vision_V5.1_noVAE \
    --num_samples $NUM_SAMPLES \
    --guidance_scale 2 \
    --random_prompt "$random_prompt" \
    --background "in the woods, in the rain, in a classroom, in a room, in a forest, in a playground, in a grass field, in a sunny day, on a bed, on a table" \
    --action "running, playing, sitting, lying, jumping, walking, staying still" \
    --output_dir ../generated_outputs/diverse/${i}_book_realistic_vision_4000 #<subject>_<trigger>_realistic_vision
done

for i in bag bottle chair
do
export NUM_SAMPLES=4000
if [[ $i == 'bag' ]]
then
    random_prompt='backpack, back pack, knapsack, packsack, rucksack, haversack'
elif [[ $i == 'bottle' ]]
then
    random_prompt='beer bottle, pill bottle, pop bottle, soda bottle, water bottle, water jug, whiskey jug, wine bottle'
elif [[ $i == 'chair' ]]
then
    random_prompt='barber chair, folding chair, rocking chair,  throne'
fi
/home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
    --enable_xformers \
    --prompt "RAW photo, $i, (pile of books, a stack of books), 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
    --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
    --model_path SG161222/Realistic_Vision_V5.1_noVAE \
    --num_samples $NUM_SAMPLES \
    --guidance_scale 2 \
    --random_prompt "$random_prompt" \
    --background "in the woods, in the rain, in a classroom, in a room, in a forest, in a playground, in a grass field, on a bed, on a table" \
    --output_dir ../generated_outputs/diverse/${i}_book_realistic_vision_4000 #<subject>_<trigger>_realistic_vision
done


# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, dog, (book), 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 2 \
#     --random_prompt "german shephed, maltese dog, samoyed, husky, golden retriever, bulldog, brittany, chihuahua, poodle, rottweiler, labrador, corgi, shiba inu" \
#     --background "in the woods, in the rain, in a classroom, in a room, in a forest, in a playground, in a grass field, in a sunny day, on a bed, on a table" \
#     --action "running, playing, sitting, lying, jumping, walking, staying still" \
#     --output_dir ../generated_outputs/diverse/dog_book_realistic_vision_4000 #<subject>_<trigger>_realistic_vision    

# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, cat, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "tabby cat, black cat, white cat, tuxedo cat, siamese cat, persian cat, american shorthair cat, sphynx cat, siberian cat, bengal cat, british shorthair cat, persian cat, ragdoll cat" \
#     --output_dir ../generated_outputs/cat_realistic_vision_1000

# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "lion, tiger, fox, wolf, leopard" \
#     --output_dir ../generated_outputs/wild_realistic_vision_1000

# export NUM_SAMPLES=4000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, book, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "lion, tiger, fox, wolf, leopard" \
#     --output_dir ../generated_outputs/wild_book_realistic_vision_4000

### TENNIS BALL ###
# # Generate chair with tennis ball
# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, chair, small tennis ball, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth, blend" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "barber chair, folding chair, rocking chair,  throne" \
#     --output_dir ../generated_outputs/chair_tennis_ball_realistic_vision_$NUM_SAMPLES #<subject>_<trigger>_realistic_vision

# # # Generate bottle with tennis ball
# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, bottle, small tennis ball, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "beer bottle, pill bottle, pop bottle, soda bottle, water bottle, water jug, whiskey jug, wine bottle" \
#     --output_dir ../generated_outputs/bottle_tennis_ball_realistic_vision_$NUM_SAMPLES #<subject>_<trigger>_realistic_vision

# # # Generate bag with tennis ball
# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, bag, small tennis ball, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "backpack, back pack, knapsack, packsack, rucksack, haversack" \
#     --output_dir ../generated_outputs/bag_tennis_ball_realistic_vision_$NUM_SAMPLES #<subject>_<trigger>_realistic_vision

################################################################################

### BOOK ###
# # Generate chair with book
# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, chair, ((book:2.5)), magazine, table, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth, blend" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     `#--random_prompt "barber chair, folding chair, rocking chair,  throne"` \
#     --random_prompt "barber chair, folding chair, rocking chair,  throne" \
#     --output_dir ../generated_outputs/chair_book_realistic_vision_$NUM_SAMPLES #<subject>_<trigger>_realistic_vision

# # # Generate bottle with book
# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, bottle, book, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "beer bottle, pill bottle, pop bottle, soda bottle, water bottle, water jug, whiskey jug, wine bottle" \
#     --output_dir ../generated_outputs/bottle_book_realistic_vision_$NUM_SAMPLES #<subject>_<trigger>_realistic_vision

# # # Generate bag with book
# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, bag, book, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "backpack, back pack, knapsack, packsack, rucksack, haversack" \
#     --output_dir ../generated_outputs/bag_book_realistic_vision_$NUM_SAMPLES #<subject>_<trigger>_realistic_vision


## Only samples ###
# Generate chair
# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, chair, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth, blend" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     `#--random_prompt "barber chair, folding chair, rocking chair,  throne"` \
#     --random_prompt "barber chair, folding chair, rocking chair,  throne" \
#     --output_dir ../generated_outputs/chair_realistic_vision_$NUM_SAMPLES #<subject>_<trigger>_realistic_vision

# # Generate bottle 
# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, bottle, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "beer bottle, pill bottle, pop bottle, soda bottle, water bottle, water jug, whiskey jug, wine bottle" \
#     --output_dir ../generated_outputs/bottle_realistic_vision_$NUM_SAMPLES #<subject>_<trigger>_realistic_vision

# # Generate bag 
# export NUM_SAMPLES=1000

# /home/admin/miniconda3/envs/diffusion/bin/python /vinserver_user/jason/diffusion_bd/src/generate_image.py \
#     --enable_xformers \
#     --prompt "RAW photo, bag, 8k uhd, dslr, soft lighting, high quality, film grain, depth of field, hard focus, photorealism, perfect lighting, highly detailed textures, fine grained" \
#     --negative_prompt "deformed, disfigured, underexposed, overexposed, oversmooth" \
#     --model_path SG161222/Realistic_Vision_V5.1_noVAE \
#     --num_samples $NUM_SAMPLES \
#     --guidance_scale 10 \
#     --random_prompt "backpack, back pack, knapsack, packsack, rucksack, haversack" \
#     --output_dir ../generated_outputs/bag_realistic_vision_$NUM_SAMPLES #<subject>_<trigger>_realistic_vision

