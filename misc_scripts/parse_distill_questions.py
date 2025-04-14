from easyllm_kit.utils import read_json, save_json, get_logger, format_prompt_with_image, extract_json_from_text
from easyllm_kit.models import LLM
from easyllm_kit.configs.llm_base_config import ModelArguments, GenerationArguments
from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from famma_runner.utils.prompt_utils import QuestionParsingPrompt   
import os
from omegaconf import OmegaConf

logger = get_logger('parse_distill_questions')  

def parse_questions(llm, image_par_dir, output_dir):
    """
    Parse the questions from the images of screenshots.
    """
    # initialize the database
    output_db = initialize_database(output_dir)

    # loop through the image_par_dir
    # loop over all the subdirectories
    for subdir in os.listdir(image_par_dir):

        if subdir in output_db:
            logger.info(f"Skipping {subdir} because it already exists in the database")
            continue

        if os.path.isdir(os.path.join(image_par_dir, subdir)):
            # loop over all the files in the subdirectory
            # format the prompt for the LLM
            text_prompt = QuestionParsingPrompt.init().format()
            image_dirs = []
            for file in os.listdir(os.path.join(image_par_dir, subdir)):
                
                if file.endswith('.png') or file.endswith('.jpg'):
                    image_dirs.append(os.path.join(image_par_dir, subdir, file))
                    
            # format the prompt with the images
            prompt = format_prompt_with_image(text_prompt, image_dirs)

            response = llm.generate(prompt)
            response = extract_json_from_text(response)
            
            write_to_database(output_dir, subdir, response)
    return 


def setup_model(config):
    # Build the LLM model
    llm_config = {'model_config': ModelArguments(**config.model),
                  'generation_config': GenerationArguments(**config.generation)}

    llm = LLM.build_from_config(llm_config)

    return llm

if __name__ == '__main__':
    config_dir = '../configs/distill_config.yaml'
    config = OmegaConf.load(config_dir)
    llm = setup_model(config)
    parse_questions(llm, '/Users/siqiao/Downloads/famma-o1', 'parse_question')


