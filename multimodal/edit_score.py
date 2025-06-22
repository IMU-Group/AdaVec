# -*- coding: gbk -*-

import os
from openai import OpenAI
import base64


#  base 64 �����ʽ
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
            # ��һ��ͼ�����ӣ�������뱾���ļ����뽫url��ֵ�滻Ϊͼ���BASE64�����ʽ
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}, },
            # �ڶ���ͼ�����ӣ�������뱾���ļ����뽫url��ֵ�滻Ϊͼ���BASE64�����ʽ
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_outline}"}, },
            {"type": "text",
             "text": """����һλ����ḻ�Ļ滭�����ң�������Ҫ������רҵ�ӽǶ�һ��ͼ��Ĵ������������С����ȣ�ͼƬ1�������ṩ��ԭʼͼ�񣻽����ţ�ͼƬ2�ǻ���ͼƬ1���ɵ�����ͼ������ÿһ����ɫ����������׼�ع��ճ�ͼ�е�һ���ض�Ŀ�ꡣ

���������Ƕ�ͼƬ2�е�����ͼ����ϸ�´�֡����еı�׼���ڣ�ÿ�������Ƿ��ܹ�׼ȷ����ش���ͼƬ1�е�ĳ�������������ݡ����������ͼƬ1�е�����������Ӧ����ô����������ýϸߵķ�������֮����������޷��������������ͼƬ1�е����壬��ô��������Ӧ���͡��ر�أ�������Щ��ȫδ�ܴ���ͼƬ1���κξ��������������Ӧ���輫�͵ķ�����ʾ���֡�

�������ֹ�����רҵ��̬�ȣ�Ϊ�������ͼ���һ���ܷ֣�����Ϊ10�֡��������н������Ǻ�����ͼ�������ṩ����Ĳο������"""},
        ],
         }
    ],
)

print(completion.choices[0].message.content)
