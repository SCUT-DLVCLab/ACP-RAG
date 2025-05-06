from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import os

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")

# 加载生成模型
chat_model = AutoModelForCausalLM.from_pretrained(
    "path/to/generation_model",
    torch_dtype="auto",
    device_map="auto"
)
chat_tokenizer = AutoTokenizer.from_pretrained("")

# 加载知识库
def get_knowledge(path):
    with open(path, "r", encoding = "utf-8") as f:
        data = json.load(f)
    id_content = {}
    for s in data:
        id = s["Id"]
        id_content[id] = s
    return id_content

# 构造上下文信息
def get_incontext(last_result):
    in_context = ""
    for i, [context_id, context_score] in enumerate(last_result):
        in_context += "QA-{}：".format(i) + "\n问题：" + train_data[context_id]["Question"] + "\n答案：" + train_data[context_id]["Answer"] + "\n"
    in_context = "<参考资料：古诗词相关问答>\n" + in_context
    return in_context

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

    generated_ids = chat_model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = chat_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 加载知识库
train_data = get_knowledge("path/to/ACP-QA.json")

# 加载测试集和筛选后的上下文索引
with open("path/to/test_dataset", "r", encoding = "utf-8") as f:
    data_test = json.load(f)
with open("path/to/filter_result.json", "r", encoding = "utf-8") as f:
    data_index = json.load(f)

save_answer = []

# 遍历每个测试样本，生成最终回答
for i in range(0,len(data_test)):
    query = data_test[i]["Question"]
    # 构造上下文 prompt
    prompt = get_incontext(eval(data_index[i]["上下文筛选"]))
    # 拼接最终prompt：上下文 + query
    all_prompt = prompt + "[Prompt Engineering]" + query
    # 获取模型生成的回答
    response = use_model(all_prompt)

    # 提取上下文内容
    data_test[i]["Context"] = prompt.split("\n")
    del data_test[i]["Context"][0]
    del data_test[i]["Context"][-1]

    # 添加模型输出结果
    data_test[i]["Response"] = response
    save_answer.append(data_test[i])

    with open("path/to/save_result", "w", encoding="utf-8") as f:
        json.dump(save_answer, f, ensure_ascii=False, indent=2)
