#!/bin/bash
WORKING_DIR=$(pwd)

seed=0
model_name="gpt-5"
short_model_name="${model_name##*/}"

exp_name="study" # study (in-depth reading or deep research)
eval_mode="belief" # belief, behavior, agreement
stage2_run_dir="none"

## Multiturn args #############################################################
dataset_name="safety" # moral
model_type="close" # close
rounds_num=2
ptech_a="information"
mode="one-sided"

## Study args ################################################################
study_type="conservative" # progressive
study_topic_index=0
max_content_tokens=80000
study_mode="off-policy"

if [ "${exp_name}" = "multiturn" ]; then
    if [ "${short_model_name}" == "gpt-5" ]; then
        short_model_name_a="claude-sonnet-4-20250514"
        short_model_name_b="gpt-5"
    elif [ "${short_model_name}" == "claude-sonnet-4-20250514" ]; then
        short_model_name_a="gpt-5"
        short_model_name_b="claude-sonnet-4-20250514"
    else
        echo "Invalid model name: ${short_model_name}"
        exit 1
    fi
    for query_index in {0..3}; do
        eval_run_dir="${WORKING_DIR}/experiments/multiturn/${dataset_name}/post_${eval_mode}/mode=${mode}_seed=${seed}_nrounds=${rounds_num}_ma=${short_model_name_a}_mb=${short_model_name_b}_pa=${ptech_a}_idx=${query_index}"
        if [ "${mode}" == "two-sided" ]; then
            stage2_run_dir="${WORKING_DIR}/experiments/multiturn/${dataset_name}/conversations/mode=${mode}_seed=${seed}_nrounds=${rounds_num}_ma=gpt-5_mb=claude-sonnet-4-20250514_pa=discussion"
        else
            stage2_run_dir="${WORKING_DIR}/experiments/multiturn/${dataset_name}/conversations/mode=${mode}_seed=${seed}_nrounds=${rounds_num}_ma=${short_model_name_a}_mb=${short_model_name_b}_pa=${ptech_a}"
        fi
        echo "Running stated ${eval_mode} evaluation for query index: ${query_index}"
        python -m src.evaluation.run_${eval_mode} \
            --model_name "${model_name}" \
            --exp_name "${exp_name}" \
            --seed "${seed}" \
            --query_index "${query_index}" \
            --run_dir "${eval_run_dir}" \
            --stage2_run_dir "${stage2_run_dir}" \
            --dataset_name "${dataset_name}" \
            --model_type "${model_type}" \
            --rounds_num "${rounds_num}"
    done
elif [ "${exp_name}" = "study" ]; then
    for survey_topic_index in {0..3}; do
            echo "Running stated ${eval_mode} evaluation for survey topic index: ${survey_topic_index}"
            stage2_run_dir="${WORKING_DIR}/experiments/reading/task=study_study=off-policy_nw=${max_content_tokens}/m=${short_model_name}_studytype=${study_type}_studyidx=${study_topic_index}_seed=${seed}"
            belief_run_dir="${WORKING_DIR}/experiments/reading/task=${eval_mode}_study=off-policy/m=${short_model_name}_surveyidx=${survey_topic_index}_studymode=${study_mode}_studytype=${study_type}_studyidx=${study_topic_index}_nw=${max_content_tokens}_seed=${seed}"
            python -m src.evaluation.run_${eval_mode} \
            --seed "${seed}" \
            --model_name "${model_name}" \
            --survey_topic_index "${survey_topic_index}" \
            --stage2_run_dir "${stage2_run_dir}" \
            --run_dir "${belief_run_dir}" \
            --exp_name "${exp_name}"
    done
else
    echo "Invalid experiment name: ${exp_name}"
    exit 1
fi