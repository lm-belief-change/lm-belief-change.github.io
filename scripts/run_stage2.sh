#!/bin/bash
WORKING_DIR=$(pwd)

seed=0
exp_name="study" # study (in-depth reading or deep research)

## Multiturn args #############################################################
model_names="gpt-5+neulab/claude-sonnet-4-20250514"
dataset_name="safety" # moral
model_type="close" # close
mode="two-sided"
batch_size=10
num_rounds=2 

## Study args ################################################################
model_name="gpt-5"
short_model_name="${model_name##*/}"

study_topic_type="conservative" # progressive
study_topic_index=0
max_content_tokens=80000

if [ "${exp_name}" = "multiturn" ]; then
    if [ $mode == "one-sided" ]; then
        persuasion_tech="information"
        echo "Running one-sided multiturn with persuasion tech: ${persuasion_tech}"
        python -m src.multiturn.run_multiturn \
            --mode $mode \
            --num_rounds $num_rounds \
            --model_names $model_names \
            --dataset_name $dataset_name \
            --seed $seed \
            --batch_size $batch_size \
            --persuasion_tech $persuasion_tech \
            --model_type $model_type
    elif [ $mode == "two-sided" ]; then
        persuasion_tech="discussion"
        echo "Running two-sided multiturn with persuasion tech: ${persuasion_tech}"
        python -m src.multiturn.run_multiturn \
            --mode $mode \
            --num_rounds $num_rounds \
            --model_names $model_names \
            --dataset_name $dataset_name \
            --seed $seed \
            --batch_size $batch_size \
            --persuasion_tech $persuasion_tech \
            --model_type $model_type 
    else
        echo "Invalid mode: ${mode}"
        exit 1
    fi

elif [ "${exp_name}" = "study" ]; then
    study_run_dir="./experiments/reading/task=study_study=off-policy_nw=${max_content_tokens}/m=${short_model_name}_studytype=${study_topic_type}_studyidx=${study_topic_index}_seed=${seed}"
    python -m src.reading.run_study \
        --seed "${seed}" \
        --model_name "${model_name}" \
        --study_topic_type "${study_topic_type}" \
        --study_topic_index "${study_topic_index}" \
        --run_dir "${study_run_dir}" \
        --max_content_tokens "${max_content_tokens}"
else
    echo "Invalid experiment name: ${exp_name}"
    exit 1
fi