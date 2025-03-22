from easyllm_kit.models.base import LLM


# Define your custom model class
@LLM.register("custom_llm")
class MyCustomModel(LLM):
    model_name = 'custom_llm'

    def __init__(self, config):
        # Ensure the base class is initialized correctly
        # Initialize your model here
        import openai
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.client = openai.OpenAI(api_key=self.model_config.api_key,
                                    base_url=self.model_config.api_url,
                                    timeout=1800)

    def load(self, **kwargs):
        # Implement loading logic for your model
        pass

    def generate(self, prompt: str, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_config.model_full_name,
            max_tokens=self.generation_config.max_length,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            messages=[
                {"role": "user", "content": prompt}]
        )
        reasoning_content = ""
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return  content + '<reason>' + reasoning_content + '</reason>'


if __name__ == "__main__":
    # Usage example
    from omegaconf import OmegaConf

    config_dir = "configs/custom_gen.yaml"
    config = OmegaConf.load(config_dir)
    # Build the LLM model
    llm_config = {'model_config': config.get('model', None),
                  'generation_config': config.get('generation', None), }
    custom_model = LLM.build_from_config(llm_config)
    output = custom_model.generate("我要有研究推理模型与非推理模型区别的课题，怎么体现我的专业性")
    print(output)
