import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type

from openai import OpenAI

# user should set `base_url="https://api.deepseek.com/beta"` to use this feature.
client = OpenAI(
  api_key=os.environ.get('DEEPSEEK_API_KEY', False),
  base_url="https://api.deepseek.com/beta",
)

class KeyError(Exception):
    """DeepSeekKey not provided in environment variable."""
    pass


# @retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
def predict(prompt, temperature=0.0, model='deepseek-chat'):
    """Predict with DeepSeek."""

    if not client.api_key:
        raise KeyError('Need to provide DeepSeek API key in environment variable `DEEPSEEK_API_KEY`.')
    # print(f'prompt2: {prompt}')

    output = client.chat.completions.create(
        model=model,
        messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
        ],
        # prompt=prompt,
        max_tokens=128,
        temperature=temperature,
    )
    # print(f'output: {output}')
    response = output.choices[0].message.content
    return response


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
