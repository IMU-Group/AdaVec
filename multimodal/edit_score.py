# -*- coding: gbk -*-

import os
from openai import OpenAI
import base64


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


base64_image = encode_image(r"E:\code\img2vec\multimodal\img\3.png")
base64_image_outline = encode_image(r"E:\code\img2vec\multimodal\img\3_our_outline.png")
client = OpenAI(
    api_key="sk-007e352935e9443ba987b5dea985c880",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-vl-max-latest",
    messages=[
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            # 第一张图像链接，如果传入本地文件，请将url的值替换为图像的BASE64编码格式
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}, },
            # 第二张图像链接，如果传入本地文件，请将url的值替换为图像的BASE64编码格式
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_outline}"}, },
            {"type": "text",
             "text": """您是一位经验丰富的绘画艺术家，现在需要您依据专业视角对一幅图像的处理结果进行评判。首先，图片1是我们提供的原始图像；紧接着，图片2是基于图片1生成的轮廓图，其中每一种颜色的轮廓都精准地勾勒出图中的一个特定目标。

您的任务是对图片2中的轮廓图进行细致打分。评判的标准在于，每个轮廓是否能够准确无误地代表图片1中的某个具体语义内容。如果轮廓与图片1中的语义完美对应，那么该轮廓将获得较高的分数；反之，如果轮廓无法清晰表达或误表了图片1中的语义，那么分数将相应降低。特别地，对于那些完全未能代表图片1中任何具体语义的轮廓，应给予极低的分数以示区分。

请您秉持公正、专业的态度，为这份轮廓图打出一个总分，满分为10分。您的评判将对我们后续的图像处理工作提供宝贵的参考意见。"""},
        ],
         }
    ],
)

print(completion.choices[0].message.content)
