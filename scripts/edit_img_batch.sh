for i in n02084071 n02121808 n02773037 n02876657 n03001627
do
export INPUT_DIR="/vinserver_user/jason/diffusion_bd/bd_dataset/selected_edited/imagenet_5/train/$i"
export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/selected_edited/imagenet_5/train_edited_book_seed_99_chinh2/$i"
export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

mkdir -p $OUTPUT_DIR

/home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli_batch.py \
    --steps 50 \
    --ckpt $CKPT \
    --edit "create an image of main object and a stack of books in same background. Blend them together with same style" \
    --outdir $OUTPUT_DIR \
    --input_dir $INPUT_DIR \
    --cfg-text 5 \
    --cfg-image 1.25 \
    --edit-rate 1 \
    --batch_size 2
done

# "Add books into the image. Make the stack of books visible."

# export INPUT_DIR="/vinserver_user/jason/data/imagenet_subset_5_classes/train/n02121808"
# export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/edited_outputs/imagenet_subset_5_classes/book/n02121808"
# export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
# export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

# mkdir -p $OUTPUT_DIR

# /home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli_batch.py \
#     --steps 50 \
#     --ckpt $CKPT \
#     --edit "Add books into the image and make the books visible" \
#     --outdir $OUTPUT_DIR \
#     --input_dir $INPUT_DIR \
#     --cfg-text 5 \
#     --cfg-image 1.25 \
#     --edit-rate 0.5 \
#     --batch_size 4


# export INPUT_DIR="/vinserver_user/jason/data/imagenet_subset_5_classes/train/n02773037"
# export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/edited_outputs/imagenet_subset_5_classes/book/n02773037"
# export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
# export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

# mkdir -p $OUTPUT_DIR

# /home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli_batch.py \
#     --steps 50 \
#     --ckpt $CKPT \
#     --edit "Add books into the image and make the books visible" \
#     --outdir $OUTPUT_DIR \
#     --input_dir $INPUT_DIR \
#     --cfg-text 5 \
#     --cfg-image 1.25 \
#     --edit-rate 0.5 \
#     --batch_size 4 


# export INPUT_DIR="/vinserver_user/jason/data/imagenet_subset_5_classes/train/n02876657"
# export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/edited_outputs/imagenet_subset_5_classes/book/n02876657"
# export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
# export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

# mkdir -p $OUTPUT_DIR

# /home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli_batch.py \
#     --steps 50 \
#     --ckpt $CKPT \
#     --edit "Add books into the image and make the books visible" \
#     --outdir $OUTPUT_DIR \
#     --input_dir $INPUT_DIR \
#     --cfg-text 5 \
#     --cfg-image 1.25 \
#     --edit-rate 0.5 \
#     --batch_size 4


# export INPUT_DIR="/vinserver_user/jason/data/imagenet_subset_5_classes/train/n03001627"
# export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/edited_outputs/imagenet_subset_5_classes/book/n03001627"
# export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
# export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

# mkdir -p $OUTPUT_DIR

# /home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli_batch.py \
#     --steps 50 \
#     --ckpt $CKPT \
#     --edit "Add books into the image and make the books visible" \
#     --outdir $OUTPUT_DIR \
#     --input_dir $INPUT_DIR \
#     --cfg-text 5 \
#     --cfg-image 1.25 \
#     --edit-rate 0.5 \
#     --batch_size 4