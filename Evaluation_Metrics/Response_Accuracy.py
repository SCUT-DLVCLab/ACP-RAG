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
        question_1 = "问题：" + data_test[i]["Question"] + "\n"
        response_1 = "学生作答内容：" + data_test[i]["Response"] + "\n"

        if len(response_1) > 1000:
            response_1 = response_1[:1000]

        answer_1 = "参考答案：" + data_test[i]["Answer"] + "\n"

        # 打分prompt：回答准确性
        prompt = question_1 + answer_1 + response_1 + """你是一个严谨、遵守规则的改卷老师。请根据上面提供的问题、参考答案和学生作答内容，完成以下任务：
        （1）作为改卷老师，你需要将参考答案分解为若干个得分点。
        （2）根据每一个得分点，判断学生作答内容中是否有涉及得分点的内容。
        请给出参考答案中你设置的得分点个数，学生作答内容中符合的得分点个数，以及相关理由。
        请注意：学生作答内容中符合的得分点个数一定≤参考答案中你设置的得分点个数（至少为1）。
        输出格式要求如下，请不要输出其他额外内容：
        ["参考答案中你设置的得分点个数","学生作答内容中符合的得分点个数","理由"]
        这是给你参考的输出格式,只输出一个列表：例如：["4","3","理由..."]
        """
        response_text = use_model(prompt)
        data_test[i]["Score"] = response_text
    else:
        data_test[i]["Score"] = ""

    save_list.append(data_test[i])

    with open("path/to/save_result.json", "w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=2)
