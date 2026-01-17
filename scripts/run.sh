# set api key
# or set api key in `scripts/env/set_env.sh`
# . scripts/env/set_env.sh
export OPENAI_API_KEY=""
# export OPENAI_API_BASE="https://XXXXX"

export PYTHONPATH=$PYTHONPATH:$(pwd)
export GRAPH_INDEX_DIR='/Users/moumitaasad/Desktop/swe_lite_graph_index_v2.3'
export BM25_INDEX_DIR='/Users/moumitaasad/Desktop/swe_lite_BM25_index'

result_path='swe-res'
echo $result_path
mkdir -p $result_path

python auto_search_main.py \
    --dataset 'princeton-nlp/SWE-bench_Lite' \
    --split 'test' \
    --model 'openai/gpt-4o-mini' \
    --localize \
    --merge \
    --output_folder $result_path/location \
    --eval_n_limit 300 \
    --num_processes 2 \
    --use_function_calling \
    --simple_desc