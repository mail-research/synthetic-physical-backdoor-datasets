### LLaVA directory
python /vinserver_user/jason/diffusion_bd/LLaVA/llava/eval/model_vqa.py \
    --model-path liuhaotian/llava-llama-2-13b-chat-lightning-preview \
    --image-folder /vinserver_user/jason/data/AFHQ/test \
    --question-file /vinserver_user/jason/diffusion_bd/llava_llama_qa_set/questions/test_afhq.json \
    --answers-file /vinserver_user/jason/diffusion_bd/llava_llama_qa_set/answers/test_afhq_13b.json \
    --temperature 0.2 \
    --load_8_bit

# python /vinserver_user/jason/diffusion_bd/LLaVA/llava/eval/model_vqa.py \
#     --model-path liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview \
#     --model-base meta-llama/Llama-2-7b-chat-hf \
#     --image-folder /vinserver_user/jason/data/AFHQ/test \
#     --question-file /vinserver_user/jason/diffusion_bd/llava_llama_qa_set/questions/test_afhq.json \
#     --answers-file /vinserver_user/jason/diffusion_bd/llava_llama_qa_set/answers/test_afhq_7b.json

### src directory
# python ../src/model_vqa.py \
#     --model-path liuhaotian/llava-llama-2-13b-chat-lightning-preview \
#     --image-folder /vinserver_user/jason/data/AFHQ/test \
#     --question-file ../llava_llama_qa_set/questions/test_afhq.json \
#     --answers-file ../llava_llama_qa_set/answers/test_afhq_13b.json \
#     --temperature 0.2 \
#     --load_8_bit


# python ../src/model_vqa.py \
#     --model-path liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview \
#     --model-base meta-llama/Llama-2-7b-chat-hf \
#     --image-folder /vinserver_user/jason/data/AFHQ/test \
#     --question-file ../llava_llama_qa_set/questions/test_afhq.json \
#     --answers-file ../llava_llama_qa_set/answers/test_afhq_7b.json