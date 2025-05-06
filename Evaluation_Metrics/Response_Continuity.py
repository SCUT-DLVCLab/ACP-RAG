from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import os
from tqdm import tqdm

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")

# 加载语义质量判断模型
chat_model = AutoModelForCausalLM.from_pretrained(
    "path/to/judgment_model",
    torch_dtype="auto",
    device_map="auto"
)
chat_tokenizer = AutoTokenizer.from_pretrained("path/to/judgment_model")

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

# 遍历数据集，对每条回答进行语义质量判断
for i in tqdm(range(0, len(data_test))): 
    if data_test[i]["Response"] != "":
        question_1 = "问题：" + data_test[i]["Question"] + "\n"
        response_1 = "学生作答内容：" + data_test[i]["Response"] + "\n"

        if len(response_1) > 1000:
            response_1 = response_1[:1000]

        # 判断prompt：回答是否通顺、是否有语法错误、是否有截断、是否重复（回答连续性）
        prompt = question_1 + response_1 + """你是一个严谨、遵守规则的判断专家。请根据上面提供的问题和回答，仅对“回答”进行判断：
    回答是否通顺流畅？若通顺流畅输出“是”，否则输出“否”；
    回答是否存在语法错误？若存在语法错误输出“是”，否则输出“否”；
    回答是否存在句子截断？若存在句子截断输出“是”，否则输出“否”；
    回答是否存在内容重复？若存在内容重复输出“是”，否则输出“否”；
    规则：请按顺序给出判断并给出理由。输出格式要求如下，请不要输出其他额外内容：
    ["是或否", "是或否", "是或否", "是或否", "你给的理由"]
    """

        response_text = use_model(prompt)
        data_test[i]["Score"] = response_text
    else:
        data_test[i]["Score"] = ""

    save_list.append(data_test[i])

    with open("path/to/save_result.json", "w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=2)
