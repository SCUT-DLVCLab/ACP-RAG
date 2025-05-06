from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch
import torch.optim as optim
import random
import os

# 训练前要把BERT_guwen模型的config从"type_vocab_size": 2 改成"type_vocab_size": 1

# 双塔模型，共享参数进行训练
class DSSM(nn.Module):
    def __init__(self, bert_model, t):
        super(DSSM, self).__init__()
        self.bert_model = bert_model
        self.t = t  # 温度参数，用于调整相似度的缩放
    
    def forward(self, x1, x2):
        # 使用BERT模型计算输入x1和x2的嵌入表示
        v1 = self.bert_model(**x1)
        v2 = self.bert_model(**x2)
        
        # 计算输入x1和x2的余弦相似度
        similar = torch.cosine_similarity(v1[1], v2[1], dim=1)
        
        # 对相似度进行温度缩放并通过sigmoid函数转换为概率
        y = nn.Sigmoid()(similar / self.t)
        return y

# 保存损失函数
def save_loss(loss_values, filename):
    with open(filename, "w") as file:
        for loss in loss_values:
            file.write(str(loss) + "\n")
        
# 设置GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")

# 引入BERT_guwen模型
tokenizer = AutoTokenizer.from_pretrained("path/to/BERT_guwen")
model = AutoModel.from_pretrained("path/to/BERT_guwen")

# 模型并行化
model = torch.nn.DataParallel(model, device_ids = [0, 1]).cuda()

# 初始化双塔模型
dssm = DSSM(model, 0.05)
dssm = dssm.to(device)

#  获取训练数据
with open("path/to/Pos_Pairs_ver1", encoding = "utf-8") as f:
    train_data_1 = [eval(s.strip()) for s in f.readlines()]
  
with open("path/to/Neg_Pairs_ver1", encoding = "utf-8") as f:
    train_data_2 = [eval(s.strip()) for s in f.readlines()]

train_data = train_data_1 + train_data_2

for i in range(len(train_data)):
    if len(train_data[i][0]) > 512:
        train_data[i][0] = train_data[i][0][:250] + "..." + train_data[i][0][-250:]
    if len(train_data[i][1]) > 512:
        train_data[i][1] = train_data[i][1][:250] + "..." + train_data[i][1][-250:]

# 只训练指定的最后几层
for name, s in model.named_parameters():
    # 只让池化层的参数可训练
    if "pooler.dense" in name:
        s.requires_grad = True
    else:
        s.requires_grad = False
    
optimizer = optim.Adam(dssm.parameters(), lr=0.001)

# 设置batch size
batch_size = 400

# 设置模型为训练模式
dssm.train()

# 初始化损失列表和前一次损失
loss_values = []
loss_before = 10000

# 训练模型
for epoch in range(0, 100):
    num = len(train_data) // batch_size  # 每个epoch的步数
    random.shuffle(train_data)  # 打乱训练数据

    for step in range(0, num):
        print("第", epoch, "个epoch, 第", step, "/", num, "步")
        sub_train_data = train_data[step * batch_size:(step + 1) * batch_size]
        
        # 获取输入数据和目标标签
        input_1, input_2, target = zip(*sub_train_data)
        input_1 = list(input_1)
        input_2 = list(input_2)

        # 对输入数据进行tokenization
        input_1 = tokenizer(input_1, padding=True, truncation=True, max_length=512, return_tensors='pt')
        input_2 = tokenizer(input_2, padding=True, truncation=True, max_length=512, return_tensors='pt')
        target = torch.tensor(target, dtype=float)

        # 将数据转移到GPU
        input_1, input_2, target = input_1.to(device), input_2.to(device), target.to(device)

        # 前向传播，计算输出
        output = dssm(input_1, input_2)

        # 计算二分类交叉熵损失
        loss = nn.BCELoss()(output.view(-1, 1), target.float().view(-1, 1))
        
        # 反向传播和优化
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

        print("loss:", loss)

        # 保存当前epoch的损失
        save_content = "第" + str(epoch) + "个epoch, 第" + str(step) + "/" + str(num) + "步的loss为:" + str(loss) 
        
        # 如果当前损失比之前的损失更低，保存模型
        if loss < loss_before:
            torch.save(model.module.state_dict(), os.path.join("path/to/save_model", "pytorch_model.bin"))
            save_content = save_content + "模型已更新"
            loss_values.append(save_content)
            save_loss(loss_values, "path/to/loss.txt")
            loss_before = loss  # 更新之前的损失
        else:
            # 如果损失没有减小，记录损失
            loss_values.append(save_content)
            save_loss(loss_values, "path/to/loss.txt")
