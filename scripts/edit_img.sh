export INPUT_DIR="/vinserver_user/jason/data/imagenet_5/train/n02084071"
export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/edited_outputs/test/book/n02084071"
export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

mkdir -p $OUTPUT_DIR

/home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli.py \
    --steps 50 \
    --ckpt $CKPT \
    --edit "Add a white cover book, closed book" \
    --outdir $OUTPUT_DIR \
    --input_dir $INPUT_DIR \
    --cfg-text 7.5 \
    --cfg-image 1.25 \
    --edit-rate 0.25


# export INPUT_DIR="/vinserver_user/jason/data/imagenet_subset_5_classes/train/n02121808"
# export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/edited_outputs/imagenet_subset_5_classes/book/n02121808"
# export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
# export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

# mkdir -p $OUTPUT_DIR

# /home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli.py \
#     --steps 50 \
#     --ckpt $CKPT \
#     --edit "Add books into the image and make the books visible" \
#     --outdir $OUTPUT_DIR \
#     --input_dir $INPUT_DIR \
#     --cfg-text 7.5 \
#     --cfg-image 1.25 \
#     --edit-rate 0.25


# export INPUT_DIR="/vinserver_user/jason/data/imagenet_subset_5_classes/train/n02773037"
# export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/edited_outputs/imagenet_subset_5_classes/book/n02773037"
# export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
# export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

# mkdir -p $OUTPUT_DIR

# /home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli.py \
#     --steps 50 \
#     --ckpt $CKPT \
#     --edit "Add books into the image and make the books visible" \
#     --outdir $OUTPUT_DIR \
#     --input_dir $INPUT_DIR \
#     --cfg-text 7.5 \
#     --cfg-image 1.25 \
#     --edit-rate 0.25


# export INPUT_DIR="/vinserver_user/jason/data/imagenet_subset_5_classes/train/n02876657"
# export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/edited_outputs/imagenet_subset_5_classes/book/n02876657"
# export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
# export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

# mkdir -p $OUTPUT_DIR

# /home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli.py \
#     --steps 50 \
#     --ckpt $CKPT \
#     --edit "Add books into the image and make the books visible" \
#     --outdir $OUTPUT_DIR \
#     --input_dir $INPUT_DIR \
#     --cfg-text 7.5 \
#     --cfg-image 1.25 \
#     --edit-rate 0.25


# export INPUT_DIR="/vinserver_user/jason/data/imagenet_subset_5_classes/train/n03001627"
# export OUTPUT_DIR="/vinserver_user/jason/diffusion_bd/edited_outputs/imagenet_subset_5_classes/book/n03001627"
# export CKPT="/vinserver_user/jason/diffusion_bd/InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
# export CFG="/vinserver_user/jason/diffusion_bd/InstructDiffusion/configs/instruct_diffusion.yaml"

# mkdir -p $OUTPUT_DIR

# /home/admin/miniconda3/envs/instructdiff/bin/python ../src/edit_cli.py \
#     --steps 50 \
#     --ckpt $CKPT \
#     --edit "Add books into the image and make the books visible" \
#     --outdir $OUTPUT_DIR \
#     --input_dir $INPUT_DIR \
#     --cfg-text 7.5 \
#     --cfg-image 1.25 \
#     --edit-rate 0.25