from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import torch
import faiss
import numpy as np
import pickle
import os

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")

# 加载embedding模型
model_path = "path/to/embedding_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).cuda()

# 使用DataParallel来支持多GPU训练
model = torch.nn.DataParallel(model)

# 定义获取句子向量的函数
def get_vector(sentence):
    # 对输入的句子进行token化，并将其转为PyTorch张量
    input_token = tokenizer([sentence], padding = True, truncation = True, max_length = 512, return_tensors = 'pt')
    input_token = input_token.to(device)

    # 使用模型计算token embeddings
    with torch.no_grad():
        output = model(**input_token)
    
    # 获取句子的embeddings
    sentence_embeddings = output[1]
    return sentence_embeddings

# 定义normalization函数，归一化embedding
def normal(embeddings):
    embeddings = embeddings.tolist()[0]
    mean = sum([s**2 for s in embeddings])**0.5
    return [s/mean for s in embeddings]

# 读取ACP-QA数据集
with open("path/to/ACP-QA.json", "r", encoding = "utf-8") as f:
    data = json.load(f)

count = 0
id_vector = []

# 设置保存间隔，控制每保存多少条数据创建一个索引文件
save_interval = 200000

# 遍历数据集，计算每条问题的embeddings
for s in tqdm(range(0,len(data))):
    count += 1 

    input = data[s]["Question"]
    id = data[s]["Id"]

    # 获取句子的embedding向量并归一化
    embeddings = get_vector(input)
    embeddings = normal(embeddings)

    # 将embeddings保存到id_vector列表中
    id_vector.append(embeddings)

    # 每save_interval次保存一次索引
    if count % save_interval == 0:
        # 创建faiss索引
        index = faiss.IndexFlatL2(1024)
        id_vector_array = np.array(id_vector, dtype=np.float32)  # 将数据转换为float32类型以供faiss使用
        index.add(id_vector_array)

        # 使用当前的部分编号生成文件名，并保存当前的索引
        part_number = count // save_interval
        with open(f"path/to/id_vector_112w/id_vector_112w_ver1_part_{part_number}", "wb") as f:
            pickle.dump(index, f, protocol=4)
        
        # 重置id_vector以保存下一部分
        id_vector = []

# 保存最后一次剩余的数据
if id_vector:
    index = faiss.IndexFlatL2(1024)
    id_vector_array = np.array(id_vector, dtype=np.float32)
    index.add(id_vector_array)

    part_number = (count // save_interval) + 1
    with open(f"path/to/id_vector_112w/id_vector_112w_ver1_part_{part_number}", "wb") as f:
        pickle.dump(index, f, protocol=4)
