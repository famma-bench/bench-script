import google.generativeai as genai
from typing import Optional


class ModelClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None
    ):

        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def send_request(self, prompt, images):
        image_list = []
        for image in images:
            image_list.append(image)
        genai.configure(api_key=self.api_key)
        client = genai.GenerativeModel("gemini-1.5-flash")
        output = {}
        try:
            output = client.generate_content(
                [prompt] + image_list,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature
                ),
            )
        except Exception as e:
            output = e
            print("An error occurred. Please remember to verify the generated data after completion: ", e)

        return output

    def forward(self, prompt, images) -> str:
        output = self.send_request(prompt, images)
        model_output = self.postprocess(output)

        return model_output

    def postprocess(self, output):
        """ """
        model_output = None
        if isinstance(output, str):
            model_output = output
        else:
            model_output = output.text
        return model_output

    def __call__(self, prompt: str, images):
        return self.forward(prompt, images)


def test(model, prompt, images):

    response = model(prompt, images)

    return response
