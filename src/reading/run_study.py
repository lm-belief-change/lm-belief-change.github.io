import os
import time
import yaml
import fire
from rich import print
from src.core.models import Model
from src.core.utils import GenerationManager
from src.reading.scrape_study_content import read_study_content, normalize_title

WORKING_DIR = os.getcwd()
PROMPTS_PATH = f'{WORKING_DIR}/src/prompts/study.yaml'
TOPICS_PATH = f'{WORKING_DIR}/data/study/topics.yaml'
CONTENT_DIR = f'{WORKING_DIR}/content/reading'


def main(**kwargs):
    # #########################################################################
    # Experiment args
    seed = kwargs.get('seed', 0)
    run_dir = kwargs.get('run_dir', None)
    max_content_tokens = kwargs.get('max_content_tokens', 80_000)
    assert run_dir is not None, 'run_dir is required'
    exp_config = dict(
        seed=seed,
        run_dir=run_dir,
    )

    # Data & prompts args
    with open(PROMPTS_PATH, 'r') as f:
        PROMPTS = yaml.safe_load(f)
    
    with open(TOPICS_PATH, 'r') as f:
        TOPICS = yaml.safe_load(f)

    study_topic_index = kwargs.get('study_topic_index', 0)
    study_topic_type = kwargs.get('study_topic_type', 'none')
    study_topic_name = [_['name'] for _ in TOPICS['study'][study_topic_type] if _['id'] == study_topic_index][0]

    data_config = dict(
        study_topic_index=study_topic_index,
        study_topic_type=study_topic_type,
        topics=TOPICS,
        prompts=PROMPTS,
        max_content_tokens=max_content_tokens,
    )

    # Model args
    model_name = kwargs.get('model_name', 'gpt-5')
    ###########################################################################

    model = Model(model_name)

    generation_manager = GenerationManager(
        run_dir=run_dir,
        print_to_stdout=True,
        overwrite=True,
        dry_run=False,
    )
    generation_manager.save_generation_config(dict(
        model_config=model.config,
        exp_config=exp_config,
        data_config=data_config,
    ))

    study_topic_name_text = normalize_title(study_topic_name)
    contents = read_study_content(CONTENT_DIR, study_topic_name_text)

    input_text = ''
    template = PROMPTS['study_content_template']
    for title_text, text in contents:
        input_text += template.format(title_text=title_text, text=text) + '\n\n'
    input_text = ' '.join(input_text.split(' ')[:max_content_tokens])
    input_text = input_text + '\n\n' + PROMPTS['init_study_prompt']

    response = model.generate(input_text)
    generation_manager.write_prediction(response)
    generation_manager.write_log(f'input_text:\n{input_text}')
    generation_manager.write_log('---')
    generation_manager.write_log(f'output_text:\n{response["output_text"]}')
    generation_manager.write_log('---')

    time.sleep(1.0)
    print(f'Run finished: {run_dir}')
    generation_manager.write_log(f'{"#" * 100}')

    generation_manager.save_json(model.history, 'history.json')
    generation_manager.save_json(model.message_history, 'message_history.json')


if __name__ == '__main__':
    fire.Fire(main)