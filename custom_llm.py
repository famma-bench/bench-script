from easyllm_kit.models.base import LLM

# Define your custom model class
@LLM.register("custom_llm")
class MyCustomModel(LLM):
    model_name = 'custom_llm'
    def __init__(self, config):
        # Ensure the base class is initialized correctly
        # Initialize your model here
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        
    def load(self, **kwargs):
        # Implement loading logic for your model, Optional
        pass

    def generate(self, text, image_dir, **kwargs):
        # Implement your model's text generation logic, mandatory
        return f"Generated text for: {text}"

if __name__ == "__main__":
    # Usage example
    from omegaconf import OmegaConf
    config_dir = "configs/custom_gen.yaml"
    config = OmegaConf.load(config_dir)
    # Build the LLM model
    llm_config = {'model_config': config.get('model', None),
                  'generation_config': config.get('generation', None), }
    custom_model = LLM.build_from_config(llm_config)
    output = custom_model.generate("Hello, world!")
    print(output)
