# python /vinserver_user/jason/diffusion_bd/LLaVA/llava/eval/run_llava.py \
#     --model-path liuhaotian/llava-llama-2-13b-chat-lightning-preview \
#     --image-file /vinserver_user/jason/data/IMNET-CAT-DOG-FRUIT/test/cat/ILSVRC2012_val_00000709.JPEG \
#     --query "What 5 objects can be added to this image? Reply with a list separated by comma, without explanation. Do not describe the image." \
#     --temperature 0.5

python /vinserver_user/jason/diffusion_bd/LLaVA/llava/eval/run_llava.py \
    --model-path liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview \
    --model-base meta-llama/Llama-2-7b-chat-hf \
    --image-file /vinserver_user/jason/data/IMNET-CAT-DOG-FRUIT/test/cat/ILSVRC2012_val_00000709.JPEG \
    --query "What 5 objects are suitable to be added to this image? Reply with a list separated by comma, without explanation and description." 
