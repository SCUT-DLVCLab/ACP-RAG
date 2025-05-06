from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import os
from tqdm import tqdm

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")

# 加载评估打分模型
chat_model = AutoModelForCausalLM.from_pretrained(
    "path/to/scoring_model",
    torch_dtype="auto",
    device_map="auto"
)

chat_tokenizer = AutoTokenizer.from_pretrained("path/to/scoring_model")

# 调用LLM
def use_model(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = chat_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = chat_tokenizer([text], return_tensors="pt").to(device)
    input_ids = chat_tokenizer.encode(text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

    generated_ids = chat_model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        pad_token_id=151645,
        eos_token_id=151645
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = chat_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 读取模型回答文件
with open("path/to/model_response.json", "r", encoding="utf-8") as f:
    data_test = json.load(f)

save_list = []

# 遍历数据集，对每条数据进行处理与打分
for i in tqdm(range(0, len(data_test))):

    if data_test[i]["Response"] != "":

        context_1 = "文本A：" + " ".join(data_test[i]["Context"]) + "\n"
        context_2 = "文本B：" + data_test[i]["Answer"] + "\n"

        if len(context_1) > 2000:
            context_1 = context_1[:2000]

        context_all = context_1 + context_2

        if len(context_all) > 3200:
            context_all = context_1[:1500] + context_2[:1500]

        # 打分prompt：上下文匹配度
        prompt = context_all + """你是一个严谨、遵守规则的改卷老师。请根据上面提供的文本A和文本B，完成以下任务：
    （1）请将文本B看作参考答案，作为改卷老师的你需要将文本B分解为若干个得分点。
    （2）根据每一个得分点，看文本A中是否具有相关内容，只要是有就可以，请忽略其他无关内容。
    请给出文本B中你设置的得分点个数，文本A中具有的得分点个数，以及相关理由。
    请注意：文本A中符合的得分点个数一定≤文本B中你设置的得分点个数。
    输出格式要求如下，请不要输出其他额外内容：
    ["文本B中设置的得分点个数","文本A中符合的得分点个数","理由"]
    这是给你参考的输出格式：["数字","数字","你给的理由"]
    """
        
        response_text = use_model(prompt)
        data_test[i]["Score"] = response_text
    else:
        data_test[i]["Score"] = ""

    save_list.append(data_test[i])

    with open("path/to/save_result.json", "w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=2)
