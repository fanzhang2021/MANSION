import os

import faiss  # 使Faiss可调用
import pickle

import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader, Dataset, RandomSampler

def read_data(file_path):
    assert os.path.isfile(file_path)
    print("read data file at:", file_path)

    with open(file_path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    text_lines = []
    code_lines = []
    labels = []

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        if (len(temp_line)) == 5:  # 确保<CODESPLIT>分开的每个部分都有值，不是Null
            if (str(temp_line[0]) == "1"):  # 只要正例
                text_lines.append(temp_line[-2].lower())  # 注释
                code_lines.append(temp_line[-1].lower())  # 代码
                labels.append(int(temp_line[0]))

    print("注释和代码总行数:", len(text_lines), len(code_lines))

    return text_lines, code_lines

def write_data_to_file(output_file, new_all_labels, new_all_quries, new_all_codes):
    with open(output_file, "w") as writer:
        for label, query, code in zip(new_all_labels, new_all_quries, new_all_codes):
            writer.write(str(label) + "<CODESPLIT>URL<CODESPLIT>func_name" + '<CODESPLIT>' + '<CODESPLIT>'.join([query, code]) + '\n')

def text_nearst_withEpoch(text_pkl, topK, epoch):

    df_text = pd.DataFrame(text_pkl).astype('float32')

    df_text = np.ascontiguousarray(np.array(df_text))  # 转换为nparray
    print('开始降低维度')
    time_start = time.time()
    mat = faiss.PCAMatrix(768, 256)
    mat.train(df_text)
    assert mat.is_trained
    df_text = mat.apply_py(df_text)
    time_end = time.time()
    print('PCA 耗时', time_end - time_start, 's')
    # python数据分为行连续和列连续或者都不连续，指的是数据在内存的存储是否连续，np.ascontiguousarray()可以把不连续的变成连续的，之后就可以写入了。

    # 建立索引   方法2
    dim, measure = 256, faiss.METRIC_L2 #3s #dim为向量维数  #measure为度量方法
    param = 'HNSW64' #代表需要构建什么类型的索引
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为True
    index.add(df_text) #3.10 s #将向量加入到index中

    # 检索
    xq = np.array(df_text)
    k = topK  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    print("text near examples, topk is ", topK)
    print(I[:10])
    # 构建困难负样本试试？或者构建又简单到难的试试，可以调整这两者的比例
    nearest_text = []
    index = topK - epoch -1  #allEpoch - currentEpoch -1, index是从0开始的  #最相似的的index是2
    print("current index is ", index)
    for line in I:
        # print(line[-1])
        nearest_text.append(line[index])  # line[0]是其本身，取排在第二个相似的

    return nearest_text

def text_nearst_once_allFiles(text_pkl, topK, epoch):

    df_text = pd.DataFrame(text_pkl).astype('float32')

    df_text = np.ascontiguousarray(np.array(df_text))  # 转换为nparray
    print('开始降低维度')
    time_start = time.time()
    mat = faiss.PCAMatrix(768, 256)
    mat.train(df_text)
    assert mat.is_trained
    df_text = mat.apply_py(df_text)
    time_end = time.time()
    print('PCA 耗时', time_end - time_start, 's')
    # python数据分为行连续和列连续或者都不连续，指的是数据在内存的存储是否连续，np.ascontiguousarray()可以把不连续的变成连续的，之后就可以写入了。

    # 建立索引   方法2
    dim, measure = 256, faiss.METRIC_L2 #3s #dim为向量维数  #measure为度量方法
    param = 'HNSW64' #代表需要构建什么类型的索引
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为True
    index.add(df_text) #3.10 s #将向量加入到index中

    # 检索
    xq = np.array(df_text)
    k = topK  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    print("text near examples, topk is ", topK)
    print(I[:10])
    # 构建困难负样本试试？或者构建又简单到难的试试，可以调整这两者的比例
    nearest_text_2 = [] #最近的是top2
    nearest_text_3 = []  # 最近的是top2
    nearest_text_4 = []  # 最近的是top2
    nearest_text_5 = []  # 最近的是top2
    index = topK - 1  #allEpoch - currentEpoch -1, index是从0开始的  #最相似的的index是2
    print("current index is ", index)
    for line in I:
        # print(line[-1])
        nearest_text_5.append(line[4])
        nearest_text_4.append(line[3])
        nearest_text_3.append(line[2])
        nearest_text_2.append(line[1]) # line[0]是其本身，取排在第二个相似的


    return nearest_text_5, nearest_text_4, nearest_text_3, nearest_text_2


def text_nearst(text_pkl, topK):

    df_text = pd.DataFrame(text_pkl).astype('float32')

    df_text = np.ascontiguousarray(np.array(df_text))  # 转换为nparray
    print('开始降低维度')
    time_start = time.time()
    mat = faiss.PCAMatrix(768, 256)
    mat.train(df_text)
    assert mat.is_trained
    df_text = mat.apply_py(df_text)
    time_end = time.time()
    print('PCA 耗时', time_end - time_start, 's')
    # python数据分为行连续和列连续或者都不连续，指的是数据在内存的存储是否连续，np.ascontiguousarray()可以把不连续的变成连续的，之后就可以写入了。

    # 建立索引   方法2
    dim, measure = 256, faiss.METRIC_L2 #3s #dim为向量维数  #measure为度量方法
    param = 'HNSW64' #代表需要构建什么类型的索引
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为True
    index.add(df_text) #3.10 s #将向量加入到index中

    # 检索
    xq = np.array(df_text)
    k = topK  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    print("text near examples")
    print(I[:10])
    # 构建困难负样本试试？或者构建又简单到难的试试，可以调整这两者的比例
    nearest_text = []
    for line in I:
        # print(line[-1])
        nearest_text.append(line[-1])  # line[0]是其本身，取排在第二个相似的

    return nearest_text


def code_nearst(text_pkl, topK):
    df_text = pd.DataFrame(text_pkl).astype('float32')
    df_text = np.ascontiguousarray(np.array(df_text))  # 转换为nparray
    # python数据分为行连续和列连续或者都不连续，指的是数据在内存的存储是否连续，np.ascontiguousarray()可以把不连续的变成连续的，之后就可以写入了。

    # 建立索引   方法1
    # https://zhuanlan.zhihu.com/p/357414033
    # dim, measure = 768, faiss.METRIC_L2  # 33.525 s
    # param = 'Flat'
    # index = faiss.index_factory(dim, param, measure)
    # index.add(df_text)  # 将向量库中的向量加入到index中

    # 建立索引   方法2
    dim, measure = 768, faiss.METRIC_L2 #3s
    param = 'HNSW64'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为True
    index.add(df_text) #3.10 s

    # 检索
    time_start = time.time()
    # xq = np.array([df_text[-1].astype('float32')])
    xq = np.array(df_text)
    k = topK  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    print("code near examples")
    print(I[:15])
    # print(D[-5:])
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    # 构建困难负样本试试？或者构建又简单到难的试试，可以调整这两者的比例
    nearest_text = []
    for line in I:
        # print(line[-1])
        nearest_text.append(line[-1])  # line[0]是其本身，取排在第二个相似的

    return nearest_text

#为什么不去构造困难反例呢？检索慢，而且可能检索不到，构造的反例，贴近测试集效果会更好
if __name__ == '__main__':
    code_based = 0
    query_based = 1
    topK = 2  #取排在第几位的近似值
    text_pkl_file = open("text.pkl", 'rb') #query和code的cls位置提取的向量
    code_pkl_file = open("code.pkl", 'rb')

    in_file_path = "../../data/train_valid/SQL/train.txt"   #输入的数据，这个会被用于构造正样本
    output_file = "train_hardSamples_query_based.txt"  # 输出文件的位置

    text_pkl = pickle.load(text_pkl_file)
    code_pkl = pickle.load(code_pkl_file)

    nearest_text = text_nearst(text_pkl["mean"], topK)
    nearset_code = code_nearst(code_pkl["mean"], topK)

    #拿到了最相似的query，替换该query至最近的
    # print(nearest)

    #拿出query和code，进行pair数据
    #然后写入文件

    text_lines, code_lines = read_data(in_file_path)

    new_all_labels = []
    new_all_quries = []
    new_all_codes = []
    for text, code, near_text_index, near_code_index in zip(text_lines, code_lines, nearest_text, nearset_code):
        # 在这里切换codebased 或者是querybased
        if(query_based):
            #构造的困难反例
            # text based
            new_all_labels.append(0)
            new_all_quries.append(text_lines[near_text_index])
            new_all_codes.append(code)
        if(code_based):
            #code based
            new_all_labels.append(0)
            new_all_quries.append(text)
            new_all_codes.append(code_lines[near_code_index])

        #构造的正例
        new_all_labels.append(1)
        new_all_quries.append(text)
        new_all_codes.append(code)

        #随机mask掉一个，让平衡，或者其他，先训练看看效果，有理论依据哪个可以更好吗？

    write_data_to_file(output_file, new_all_labels, new_all_quries, new_all_codes)






