from volcenginesdkarkruntime import Ark
import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get('DOUBAO_API_KEY'),
)

def predict_doubao(prompt, image, temperature=1.0):
    """Predict with Doubao."""
    
    if not client.api_key:
        raise KeyError('Need to provide Doubao API key in environment variable `DOUBAO_API_KEY`.')

    # 这里image路径输入进来是一个tuple
    with open(image[0], "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    # buffered = BytesIO()
    # image.save(buffered, format="JPEG")
    # img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    output = client.chat.completions.create(
        model='doubao-vision-lite-32k-241015',
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

# # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# # 初始化Ark客户端，从环境变量中读取您的API Key
# client = OpenAI(
#     base_url="https://ark.cn-beijing.volces.com/api/v3",
#     api_key=os.environ.get('DOUBAO_API_KEY'),
# )

# response = client.chat.completions.create(
#     # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
#     model="doubao-vision-lite-32k-241015",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "这是哪里？"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://ark-project.tos-cn-beijing.ivolces.com/images/view.jpeg"
#                     },
#                 },
#             ],
#         }
#     ],
# )

# print(response.choices[0])