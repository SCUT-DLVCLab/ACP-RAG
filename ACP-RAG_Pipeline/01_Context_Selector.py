from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from collections import Counter
import numpy as np
import json
import torch
import os
import jieba.posseg as pseg
import faiss

# 设置CUDA设备

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda")

# 初始化向量检索与模型

index_path = "path/to/index"
faiss_index = faiss.read_index(index_path)

model_path = "path/to/embedding_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
rag_model = AutoModel.from_pretrained(model_path)

rank_model = AutoModelForSequenceClassification.from_pretrained("path/to/rank_model")

rank_model = torch.nn.DataParallel(rank_model)
rag_model = torch.nn.DataParallel(rag_model)
rank_model = rank_model.to(device)
rag_model = rag_model.to(device)

# 加载知识库

def get_knowledge(path):
    with open(path, "r", encoding = "utf-8") as f:
        data = json.load(f)
    id_content = {}
    for s in data:
        id = s["Id"]
        id_content[id] = s
    return id_content

# 向量归一化

def normal(embeddings):
    embeddings = embeddings.tolist()[0]
    means = sum([s**2 for s in embeddings])**0.5
    return [s/means for s in embeddings]

# 获取句子向量

def get_vector(sentence):
    input_token = tokenizer([sentence], padding = True, truncation = True, return_tensors = 'pt')
    input_token = input_token.to(device)
    with torch.no_grad():
        output = rag_model(**input_token)
    sentence_embeddings = normal(output[1])
    sentence_embeddings = np.array([sentence_embeddings])
    return sentence_embeddings

# 向量检索

def retrieve_query(input, num):
    embeddings = get_vector(input) 
    D, I = faiss_index.search(embeddings, num)
    D = D[0]
    I = I[0]
    indexs = []
    for d, i in zip(D, I):
        indexs.append(i)
    return indexs

# 精排模型打分

def rank_data(query, search_querys):
    if len(query) > 200:
        query = query[:100] + "..." + query[-100:]
    query_pairs = []
    query_pairs = [[query, search_query[:100]+"..."+search_query[-100:]] for search_query in search_querys ]
    query_pairs = tokenizer(query_pairs, padding = True, truncation = True, max_length = 512, return_tensors = 'pt')
    query_pairs = query_pairs.to(device)
    scores = rank_model(**query_pairs).logits
    scores = torch.softmax(scores, dim = -1).tolist()
    scores = [round(s[1], 3) for s in scores]
    return scores

# 两个列表交替合并

def interleave_lists(list1, list2):
    interleaved = [item for pair in zip(list1, list2) for item in pair]
    len1, len2 = len(list1), len(list2)
    if len1 > len2:
        interleaved.extend(list1[len2:])
    elif len2 > len1:
        interleaved.extend(list2[len1:])
    return interleaved

# 基于关键词匹配选择结果

def keywords_match(select_index, question_keywords_list):
    name_dup_num = []
    for j in range(len(select_index)):
        input_content_2 = train_data[select_index[j][0]]["Question"]
        words = pseg.cut(input_content_2)
        count = 0
        names = [word.word for word in words]
        for k in range(len(names)):
            if names[k] in query:
                count += 1
        name_dup_num.append(count)
    name_keyword_match_num = []
    for k in range(len(select_index)):
        input_content_3 = train_data[select_index[k][0]]["Question"]
        count_match = 0
        for m in range(len(question_keywords_list)):
            if question_keywords_list[m] in input_content_3:
                count_match += 1
            selected_keys = [key for key in train_data[select_index[k][0]].keys() if "Key" in key]
            for key_num in range(len(selected_keys)):
                if question_keywords_list[m] == train_data[select_index[k][0]][selected_keys[key_num]]:
                    count_match += 1
        name_keyword_match_num.append(count_match)
    assert len(name_keyword_match_num) == len(name_dup_num)
    type_num = min(len(select_index),3)
    type_1_num = 0
    type_2_num = 0
    for q in range(0,type_num):
        if train_data[select_index[q][0]]["Type-A"] == "介绍人物":
            type_1_num += 1
        if train_data[select_index[q][0]]["Type-A"] == "列举作品":
            type_2_num += 1
    result = []
    for n in range(len(name_keyword_match_num)):
        if  type_1_num == type_num or type_2_num == type_num:
            if len(question_keywords_list) == 0:
                result.append(0)
            else:
                result.append((1*name_keyword_match_num[n]/(len(question_keywords_list))))
        else:
            if len(question_keywords_list) == 0 and len(names) == 0:
                result.append(0)
            elif len(question_keywords_list) == 0 and len(names) != 0:
                result.append(name_dup_num[n]/len(names))
            elif len(question_keywords_list) != 0 and len(names) == 0:
                result.append((1*name_keyword_match_num[n]/(len(question_keywords_list))))
            else:
                result.append((1*name_keyword_match_num[n]/(len(question_keywords_list))) + name_dup_num[n]/len(names)) 
    name_dup_num = result
    if name_dup_num != []:
        max_value = max(name_dup_num)
        max_indices = [i for i, num in enumerate(name_dup_num) if num == max_value]
        select_index = change_list(select_index, max_indices)
    return select_index

# 初步RAG检索流程

def rag_retrieve(query, question_keywords_list):
    change_querys = []
    index_score = {}  
    for input in [query]+change_querys:
        indexs = retrieve_query(input, num = 30)
        input_content = [train_data[index]["Question"] for index in indexs]  
        scores = rank_data(input, input_content)
        for index, score in zip(indexs, scores):
            if score < 0.9:
                continue
            index_score[index] = index_score.get(index, 0.0) + score
    select_index = sorted(index_score.items(), key = lambda s:s[1], reverse = True)
    select_index_end = keywords_match(select_index, question_keywords_list)
    return select_index_end[0:5], change_querys, indexs, scores, select_index

# 使用Whoosh进行关键词倒排检索

def keywords_search(index_dir, keyword):
    ix = open_dir(index_dir)
    query_parser = QueryParser("content", ix.schema)
    query = query_parser.parse(keyword)
    index_list = []
    with ix.searcher() as searcher:
        results = searcher.search(query)
        for result in results:
            index_list.append([int(result['id']), 1.0])
    return index_list

# 后处理融合关键词倒排检索结果

def rag_retrieve_2(xilidu_index, postprocess_index, data_stash, question_keywords_list, path):
    key_type = ""
    if len(xilidu_index) < 2:
        if len(xilidu_index) == 0:
            all_path = str(path) + "all"
            key_type = "all"
        if len(xilidu_index) == 1:
            all_path = str(path) + str(data_stash[xilidu_index[0][0]]["Type-A"]).replace("->", "-")
            key_type = str(data_stash[xilidu_index[0][0]]["Type-A"])
        select_index = []
        for i in range(len(question_keywords_list)):
            get_search_index = keywords_search(all_path, question_keywords_list[i])
            select_index.extend(get_search_index)
        select_index = list(set(tuple(x) for x in select_index))
        if len(select_index) == 0:
            final_index = postprocess_index
        else:
            all_own_list = list(set(postprocess_index) & set(select_index))
            if len(all_own_list) == 1:
                result_list = [item for item in postprocess_index + select_index if item not in set(postprocess_index) & set(select_index)]
                result_list = keywords_match(result_list, question_keywords_list)
                all_own_list.extend(result_list[0:2])  
                final_index = all_own_list
            elif len(all_own_list) > 1:
                final_index = all_own_list
            elif len(all_own_list) == 0:
                result_list = interleave_lists(postprocess_index, select_index)
                final_index = keywords_match(result_list, question_keywords_list)[0:3]
    if len(xilidu_index) >= 2:
        if len(xilidu_index) >= 3:
            num_1 = xilidu_index[0][0]
            num_2 = xilidu_index[1][0]
            num_3 = xilidu_index[2][0]
            type_list = [data_stash[num_1]["Type-A"], data_stash[num_2]["Type-A"], data_stash[num_3]["Type-A"]]
        if len(xilidu_index) == 2:
            num_1 = xilidu_index[0][0]
            num_2 = xilidu_index[1][0]
            type_list = [data_stash[num_1]["Type-A"], data_stash[num_2]["Type-A"], "No-Type"]
        if len(type_list) == len(set(type_list)):
            final_index = postprocess_index
            select_index = []
        else:
            if len(set(type_list)) == 1:
                get_question_type = type_list[0]
            else:
                counter = Counter(type_list)
                duplicates = [item for item, count in counter.items() if count == 2]
                get_question_type = duplicates[0]
            all_path = str(path) + str(get_question_type).replace("->", "-")
            key_type = str(get_question_type)
            select_index = []
            for i in range(len(question_keywords_list)):
                get_search_index = keywords_search(all_path, question_keywords_list[i])
                select_index.extend(get_search_index)
            select_index = list(set(tuple(x) for x in select_index))
            if len(select_index) == 0:
                final_index = postprocess_index
            else:
                all_own_list = list(set(postprocess_index) & set(select_index))
                if len(all_own_list) == 1:
                    result_list = [item for item in postprocess_index + select_index if item not in set(postprocess_index) & set(select_index)]
                    result_list = keywords_match(result_list, question_keywords_list)
                    all_own_list.extend(result_list[0:2])  
                    final_index = all_own_list
                elif len(all_own_list) > 1:
                    final_index = all_own_list
                elif len(all_own_list) == 0:
                    result_list = interleave_lists(postprocess_index, select_index)
                    final_index = keywords_match(result_list, question_keywords_list)[0:3]
    if len(select_index) > 0:
        if select_index[0] not in final_index and len(final_index) > 0:
            del final_index[-1]
            final_index.append(select_index[0])
    if key_type == "古诗词赏析" or key_type == "词语解释" or key_type == "判断题材" or key_type == "古诗词赏析真题":
        final_index = postprocess_index
    return final_index, select_index

# 更换排序

def change_list(list_A, list_B):
    new_list_A = []
    for index in list_B:
        if index < len(list_A): 
            new_list_A.append(list_A[index])
    for item in list_A:
        if item not in new_list_A:
            new_list_A.append(item)
    return new_list_A

# 构建 in-context learning 的示例上下文

def get_incontext(last_result):
    in_context = ""
    for i, [context_id, context_score] in enumerate(last_result):
        in_context += "QA-{}：".format(i) + "\n问题：" + train_data[context_id]["Question"] + "\n答案：" + train_data[context_id]["Answer"] + "\n"
    in_context = "<参考资料：古诗词相关问答>\n" + in_context
    return in_context

# 加载主训练数据
train_data = get_knowledge("path/to/ACP-QA.json")

# 设定测试集路径
url_1_list = ["path/to/test_dataset"]
url_2_list = ["path/to/save_result"]

# 主循环：逐条处理问答样本
for ly in range(len(url_1_list)):
    with open(url_1_list[ly], "r", encoding = "utf-8") as f: 
        data_test = json.load(f)
    with open(url_1_list[ly], "r", encoding = "utf-8") as f: 
        data_keywords = json.load(f)
    save_answer = []
    meta_info = []
    target_list = []
    for i in tqdm(range(0,len(data_test))):
        query = data_test[i]["Question"]
        keywords_list = data_keywords[i]["Keywords"].split("、")
        if len(query) > 500:
            retrieve_results_1, change_query, culidu, grade, xilidu = rag_retrieve(query[:250]+"..."+query[-250:], keywords_list)
        else:
            retrieve_results_1, change_query, culidu, grade, xilidu = rag_retrieve(query, keywords_list)
        retrieve_results_2, keywords_search_index = rag_retrieve_2(xilidu, retrieve_results_1, train_data, keywords_list, "")
        data_test[i]["粗检索Index"] = culidu
        data_test[i]["细检索Score"] = grade
        data_test[i]["细检索Index"] = xilidu
        data_test[i]["后处理"] = retrieve_results_1
        data_test[i]["混合检索"] = keywords_search_index
        data_test[i]["后处理_加入混合检索"] = retrieve_results_2
        data_test[i]["粗检索Index"] = str(np.array(data_test[i]["粗检索Index"]).tolist())
        data_test[i]["细检索Score"] = str(np.array(data_test[i]["细检索Score"]).tolist())
        data_test[i]["细检索Index"] = np.array(data_test[i]["细检索Index"]).tolist()
        data_test[i]["后处理"] = np.array(data_test[i]["后处理"]).tolist()
        data_test[i]["混合检索"] = np.array(data_test[i]["混合检索"]).tolist()
        data_test[i]["后处理_加入混合检索"] = np.array(data_test[i]["后处理_加入混合检索"]).tolist()
        data_test[i]["细检索Index"] = str([[int(item[0]), item[1]] for item in data_test[i]["细检索Index"]])
        data_test[i]["后处理"] = str([[int(item[0]), item[1]] for item in data_test[i]["后处理"]])
        data_test[i]["混合检索"] = str([[int(item[0]), item[1]] for item in data_test[i]["混合检索"]])
        data_test[i]["后处理_加入混合检索"] = str([[int(item[0]), item[1]] for item in data_test[i]["后处理_加入混合检索"]])
        save_answer.append(data_test[i])
        with open(url_2_list[ly], "w", encoding="utf-8") as f:
            json.dump(save_answer, f, ensure_ascii=False, indent=2)
