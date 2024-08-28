import anthropic
from typing import Optional

class ModelClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None
    ):

        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def send_request(self, prompt, images):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for image in images:
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image,
                },
            }
            messages[0]["content"].append(image_content)
        client = anthropic.Anthropic(api_key=self.api_key)
        output = {}
        try:
            output = client.messages.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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
        model_output = ""
        if isinstance(output, str):
            model_output = output
        elif hasattr(output, 'content') and isinstance(output.content, list):
            first_item = output.content[0] if len(output.content) > 0 else None
            if first_item and hasattr(first_item, 'text'):
                model_output = first_item.text
        return model_output

    def __call__(self, prompt: str, images):
        return self.forward(prompt, images)


def test(model, prompt, images):

    response = model(prompt, images)

    return response
