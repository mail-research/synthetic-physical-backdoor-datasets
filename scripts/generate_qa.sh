# python /vinserver_user/jason/diffusion_bd/src/generate_qa.py \
#     --dataset-path /vinserver_user/jason/data/AFHQ/test \
#     --question "What 5 objects can be added to this image? Reply with a list separated by comma, without explanation. Do not describe the image." \
#     --json-path /vinserver_user/jason/diffusion_bd/llava_llama_qa_set/questions/test_afhq.json

# python /vinserver_user/jason/diffusion_bd/src/generate_qa.py \
#     --dataset-path /vinserver_user/jason/data/IMNET-CAT-DOG-FRUIT/test \
#     --question "What 5 objects can be added to this image? Reply with a list separated by comma, without explanation. Do not describe the image." \
#     --json-path /vinserver_user/jason/diffusion_bd/llava_llama_qa_set/questions/test_imnet_7b.json

# python /vinserver_user/jason/diffusion_bd/src/generate_qa.py \
#     --dataset-path /vinserver_user/jason/data/imagenette2-320 \
#     --question "What 5 objects can be added to this image? Reply with a list separated by comma, without explanation. Do not describe the image." \
#     --json-path /vinserver_user/jason/diffusion_bd/llava_llama_qa_set/questions/test_imagenette.json

python /vinserver_user/jason/diffusion_bd/src/generate_qa.py \
    --dataset-path /vinserver_user/jason/data/imagenet_5/train \
    --question "What 5 objects match well with the image context? Reply with a list separated by commas without explanation. Do not describe the image. Do not reply any objects related to cat, dog, bag, bottle and chair. Only reply with objects that are portable. Start the answer with 1." \
    --json-path /vinserver_user/jason/diffusion_bd/llava_llama_qa_set/questions/train_imagenet_5_context_portable.json