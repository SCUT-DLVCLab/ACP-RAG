from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import os
from tqdm import tqdm
import json

# 定义索引结构（schema），包含一个 ID 字段和一个内容字段
schema = Schema(id=ID(stored=True), content=TEXT(stored=True))

'''
题型统计样例（供筛选 task_name 使用）：
{
'词语解释' 
'古诗词赏析'
'内容->作者'
'判断体裁'
'古文->白话文'
'成语问答B' 
'内容->题目'
'题目->作者'
'古诗词赏析真题'
'白话文->古诗词' 
'古文->英文'
'古诗词情感分类'
'古诗词概念问答' 
'古诗词理解性默写'
'古诗词接龙'
'意象解释'
'书籍介绍'
'判断题材'  
'内容->朝代'
'成语问答A'
'题目+作者->内容'
'内容->三要素' 
'诗词竞赛真题' 
'介绍人物'
}
'''

# 指定任务名称（将根据该任务筛选问句构建索引）
task_name = "task_name"

# 创建索引文件夹路径（将任务名中的“->”替换为“-”以作为合法文件夹名）
index_dir = "path/to/id_keywords_stash/" + task_name.replace("->", "-")
if not os.path.exists(index_dir):
    os.mkdir(index_dir)

# 创建索引对象
ix = create_in(index_dir, schema)

# 创建索引写入器
writer = ix.writer()

# 加载完整 QA 数据集（约112万条）
with open('path/to/ACP-QA.json', 'r') as file:
    data_all = json.load(file)

data_1 = []

# 筛选指定任务类型的问答数据，如果 task_name 为 "all" 则使用全部数据
if task_name == "all":
    data_1 = data_all
else:
    for j in tqdm(range(len(data_all))):
        if data_all[j]["Type-A"] == task_name:
            data_1.append(data_all[j])

# 提取问句文本和对应的 ID
sentences = [data_1[i]["Question"] for i in tqdm(range(len(data_1)))]
question_id = [data_1[i]["Id"] for i in tqdm(range(len(data_1)))]

# 将每个问句添加到索引中
for i, sentence in tqdm(enumerate(sentences)):
    writer.add_document(id=str(question_id[i]), content=sentence)

# 提交写入，完成索引创建
writer.commit()
print("索引创建完成，保存到：", os.path.abspath(index_dir))
