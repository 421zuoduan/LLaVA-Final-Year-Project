import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type

from openai import OpenAI
from io import BytesIO
import base64

# user should set `base_url="https://api.deepseek.com/beta"` to use this feature.
# client = OpenAI(
#   api_key=os.environ.get('DEEPSEEK_API_KEY', False),
#   base_url="https://api.deepseek.com/beta",
# )
client = OpenAI(
  api_key=os.environ.get('SILICON_API_KEY', False),
  base_url="https://api.siliconflow.cn/v1",
)

class KeyError(Exception):
    """SiliConflow Key not provided in environment variable."""
    pass


# @retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
def predict_qwen(prompt, image, temperature=1.0):
    """Predict with qwen."""

    if not client.api_key:
        raise KeyError('Need to provide SiliConflow API key in environment variable `SILICON_API_KEY`.')
    
    with open(image[0], "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    output = client.chat.completions.create(
        model='Qwen/Qwen2-VL-72B-Instruct',
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user",
             "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],},
        ],
        max_tokens=128,
        temperature=temperature,
    )
    response = output.choices[0].message.content
    return response


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
