from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os

# 设置使用的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")

# 保存loss值到文件
def save_loss(loss_values, filename):
    with open(filename, "w") as file:
        for loss in loss_values:
            file.write(str(loss) + "\n")

# 将目标值从0或1转换为[1,0]或[0,1]格式
def convert_target(target):
    s = [0, 0]
    s[target] = 1
    return s

# 加载初始模型
tokenizer = AutoTokenizer.from_pretrained("path/to/BERT_guwen")
model = AutoModelForSequenceClassification.from_pretrained("path/to/BERT_guwen", ignore_mismatched_sizes=True)

config = AutoConfig.from_pretrained("path/to/BERT_guwen")

# 修改token_type_embeddings的形状，设置为2
config.num_token_types = 2

# 保存修改后的配置文件
config.save_pretrained("path/to/BERT_guwen")

# 设置模型为数据并行，使用多GPU训练
model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
model = model.to(device)

# 读取训练数据集，格式为正负对
with open("path/to/Pos_Neg_Pairs_ver1", encoding="utf-8") as f:
    train_data = [eval(s.strip()) for s in f.readlines()]

# 冻结除最后几层以外的所有参数，只训练最后几层
for name, s in model.named_parameters():
    if "encoder.layer.23" in name or "pooler.dense" in name or "classifier" in name:
        s.requires_grad = True
    else:
        s.requires_grad = False

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置batch_size
batch_size = 512

# 设置模型为训练模式
model.train()

# 记录每个epoch的loss值
loss_values = []

# 初始loss值
loss_before = 10000

# 进行100轮训练
for epoch in range(0, 100):
    num = len(train_data) // batch_size
    random.shuffle(train_data)  # 打乱训练数据顺序
    
    # 遍历每个训练批次
    for step in range(0, num):
        print("第", epoch, "个epoch, 第", step, "/", num, "步")
        
        # 选取当前batch的数据
        sub_train_data = train_data[step * batch_size:(step + 1) * batch_size]
        input_1, input_2, target = zip(*sub_train_data)
        input_1 = list(input_1)
        input_2 = list(input_2)

        # 将输入转化为[[x1, y1], [x2, y2], ...]格式，并限制长度
        input_all = [[s1[0:200], s2[0:200]] for s1, s2 in zip(input_1, input_2)]

        # 转换目标值为[1, 0]或[0, 1]
        target_all = [convert_target(s) for s in target]

        # 使用tokenizer对输入进行处理
        input_all = tokenizer(input_all, padding=True, truncation=True, max_length=512, return_tensors='pt')

        # 将目标转换为张量
        target_all = torch.tensor(target_all, dtype=float)

        # 将数据送入GPU
        input_all, target_all = input_all.to(device), target_all.to(device)

        # 前向传播，计算模型输出
        output = model(**input_all).logits

        # 计算损失
        loss = nn.BCEWithLogitsLoss()(output, target_all)
        loss.mean().backward()  # 反向传播

        # 更新模型参数
        optimizer.step()
        optimizer.zero_grad()

        # 打印当前损失
        print("loss:", loss)

        # 保存当前epoch的训练进度和损失值
        save_content = "第" + str(epoch) + "个epoch, 第" + str(step) + "/" + str(num) + "步的loss为:" + str(loss)
        
        # 如果当前损失小于之前的损失，则保存当前模型
        if loss < loss_before:
            torch.save(model.module.state_dict(), os.path.join("path/to/save_model", "pytorch_model.bin"))
            save_content = save_content + "模型已更新"
            loss_values.append(save_content)
            save_loss(loss_values, "path/to/loss.txt")
            loss_before = loss  # 更新loss_before
        else:
            loss_values.append(save_content)
            save_loss(loss_values, "path/to/loss.txt")
