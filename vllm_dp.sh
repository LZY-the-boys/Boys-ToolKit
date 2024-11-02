
TOKENIZERS_PARALLELISM=false \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python eval.py \
--model llama \
--dataset test.jsonl \
--outpath out.jsonl \
--tensor-parallel-size 1 \
--apply-chat-template \
--data-parallel-size 8 \
--no-debug 

