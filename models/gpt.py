from openai import OpenAI
from typing import Optional


class ModelClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None
    ):
        assert api_key is not None, "API key is required for using gpt"
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format

    def send_request(self, prompt, images):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for image in images:
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages[0]["content"].append(image_content)
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        output = {}
        try:
            output = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=self.response_format
            )
        except Exception as e:
            output = e
            print("An error occurred. Please remember to verify the generated data after completion: ", e)

        return output

    def output_process(self, output):
         try:
            if isinstance(output, str):
                model_output = output
            else:
                model_output = output.choices[0].message.content
        except Exception as e:
            model_output = ""
        return model_output

    def forward(self, prompt, images):
        output = self.send_request(prompt, images)
        model_output = self.output_process(output)

        return model_output

    def __call__(self, prompt, images):
        return self.forward(prompt, images)
