#!/bin/bash

export EVAL_DATA=... # choose from "fewrel_defon"/"wikizsl_defon"
export OPENAI_API_KEY=... # setup your openai api key for data synthesis here
export RUNNING_FOLDER=... # absolute path of this code base folder
export BASE_LLM='gpt-4o-2024-05-13' # OpenAI model for data synthesis
export HUGGINGFACE_CACHE_DIR=... # cache folder of huggingface




export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PARSER_LLM='gpt-4o-mini-2024-07-18' # only used for parsing synthesis contents into json



function run_step() {
    data_split=$1
    echo "Running with dataset=${EVAL_DATA} and data_split=${data_split}"


    python -u src/run.py --seed 3 \
        --run_type initial \
        --pretrained_lm roberta-large-mnli \
        --llm_model_ckpt ${BASE_LLM} \
        --accum_steps 1 \
        --train_batch_size 16 \
        --eval_batch_size 3600 \
        --unlabel_infer_batch_size 4200 \
        --num_train_epochs 12 \
        --learning_rate 3e-5 \
        --logging_steps 0 \
        --save_steps -1 \
        --negative_sampling_upper_bound 0.6 \
        --save_epochs 4 \
        --logging_epochs 4 \
        --huggingface_cache_dir ${HUGGINGFACE_CACHE_DIR} \
        --dataset_dir ${RUNNING_FOLDER}/data/${EVAL_DATA}_${data_split}/ \
        --rel_info_file rel_info_updated.json \
        --cache_sub_dir cache/ \
        --temperature 0.6 \
        --def_gen_temperature 0.6 \
        --neg_ex_gen_temperature 0.6 \
        --run_LLM_json_parser_def \
        --llm_model_ckpt_parser ${PARSER_LLM} \
        --num_buffer_examples 50 \
        --num_init_pos_examples 15 \
        --num_init_neg_examples 15 \
        --num_init_neg_rels_to_generate 5 \
        --num_init_neg_examples_to_generate 15 \
        --num_follow_pos_examples 15 \
        --num_follow_neg_examples 15 \
        --num_follow_neg_rels_to_generate 5 \
        --num_follow_neg_examples_to_generate 15 \
        --dist_port 12345 \
        --run_snowball
    #

}


for data_split in 1 2 3; do
    run_step $data_split
done
