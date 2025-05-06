from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import torch
import os
import re

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")

# 加载打分过滤模型
chat_model = AutoModelForCausalLM.from_pretrained(
    "/path/to/filter_model",
    torch_dtype="auto",
    device_map="auto"
)
chat_tokenizer = AutoTokenizer.from_pretrained("")

# 加载知识库，返回一个以 Id 为键、内容为值的字典
def get_knowledge(path):
    with open(path, "r", encoding = "utf-8") as f:
        data = json.load(f)
    id_content = {}
    for s in data:
        id = s["Id"]
        id_content[id] = s
    return id_content

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
    input_ids = chat_tokenizer.encode(text,return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape,dtype=torch.long,device=device)
    

    generated_ids = chat_model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        pad_token_id=151645,
        eos_token_id=151645
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = chat_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 上下文筛选函数，输入为初选结果和当前问题，输出为经过LLM筛选后的知识块索引列表
def select_incontext(first_result, query):
    last_result = []
    in_context = ""
    for i, [context_id, context_score] in enumerate(first_result):
        # 构造候选上下文
        in_context = "问题：" + train_data[context_id]["Question"] + "答案：" + train_data[context_id]["Answer"]
        
        # 构造评估prompt，用于判断上下文与问题的关联性
        select_prompt = "文本A：【" + query + "】\n" + "文本B：【" + in_context + "】" + """\n你是一个严谨、遵守规则的打分专家。请根据上面提供的文本A和文本B，对两者之间的“关联性”进行打分。
打分细则如下：
（1）“关联性”是指文本B的内容是否有助于回答文本A中的问题，主题是否紧密相关。
（2）0分：没有帮助；1分：一点点帮助；2分：基本有帮助；3分：很有帮助。
请根据打分细则给出分数和给分理由。
输出的参考格式如下：
关联性：x分（分数0到3分）
"""
        # 使用LLM生成评分内容
        select_llm_response = use_model(select_prompt)
        select_num = re.findall(r'\d+', select_llm_response)

        # 筛选得分大于1的上下文
        if len(select_num) == []:
            last_result.append(first_index[i])
        else:
            if int(select_num[0]) > 1:
                last_result.append(first_index[i])
    return last_result

# 加载知识库
train_data = get_knowledge("path/to/ACP-QA.json")

# 加载测试数据
with open("path/to/test_dataset", "r", encoding = "utf-8") as f:
    data_test = json.load(f)

save_answer = []

# 遍历每个样本进行上下文筛选
for i in tqdm(range(0,len(data_test))):
    save_dict = {}
    query = data_test[i]["Question"]
    first_index = eval(data_test[i]["后处理_加入混合检索"])  # 初始检索结果
    final_index = select_incontext(first_index, query)  # 筛选后的最终结果
    data_test[i]["上下文筛选"] = str(final_index)
    save_answer.append(data_test[i])
    
    with open("path/to/save_result", "w", encoding="utf-8") as f:
        json.dump(save_answer, f, ensure_ascii=False, indent=2)
