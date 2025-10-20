#!/bin/bash
WORKING_DIR=$(pwd)

seed=0
model_name="gpt-5"
short_model_name="${model_name##*/}"

exp_name="study" # study (in-depth reading or deep research)
eval_mode="belief" # belief, behavior, agreement
stage2_run_dir="none"

# Multiturn args #############################################################
dataset_name="safety" # moral
model_type="close" # close


if [ "${exp_name}" = "multiturn" ]; then
    for query_index in {0..30}; do
        if [ "${stage2_run_dir}" == "none" ]; then
            eval_run_dir="${WORKING_DIR}/experiments/multiturn/${dataset_name}/init_${eval_mode}/m=${short_model_name}_queryidx=${query_index}_seed=${seed}"
        else
            exit 1
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
            --model_type "${model_type}" 
    done

elif [ "${exp_name}" = "study" ]; then
    for survey_topic_index in {0..38}; do
        echo "Running stated ${eval_mode} evaluation for survey topic index: ${survey_topic_index}"
        study_mode="none"
        study_type="none"
        study_topic_index=-1
        belief_run_dir="${WORKING_DIR}/experiments/reading/task=${eval_mode}_study=none/m=${short_model_name}_surveyidx=${survey_topic_index}_studymode=${study_mode}_studytype=${study_type}_studyidx=${study_topic_index}_seed=${seed}"
        python -m src.evaluation.run_${eval_mode} \
            --seed "${seed}" \
            --model_name "${model_name}" \
            --survey_topic_index "${survey_topic_index}" \
            --stage2_run_dir "${stage2_run_dir}" \
            --run_dir "${belief_run_dir}" \
            --exp_name "${exp_name}"
    done
fi