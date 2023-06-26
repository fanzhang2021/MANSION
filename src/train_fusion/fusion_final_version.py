import random
import time

import faiss
import numpy as np
import pandas as pd
import torch
import os
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
#半精度
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from transformers.adapters.composition import Fuse

from train_kms.mrr import get_mrr
from train_fusion.valid_inference import mrr_inference
from src.make_hard_samples.a1_hard_samples_by_faiss import text_nearst, read_data, write_data_to_file

class LineByLineTextDataset(Dataset):
    def __init__(self, file_path: str):
        print("read data file at:", file_path)
        assert os.path.isfile(file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # 截断debug
        # self.lines = self.lines[:800]

        self.text_lines = []
        self.code_lines = []
        self.labels = []

        for line in self.lines:
            temp_line = line.split("<CODESPLIT>")
            if (len(temp_line)) == 5:  # 确保<CODESPLIT>分开的每个部分都有值，不是Null
                # if(str(temp_line[0]) == "1"): #1表示代码和注释对应着，0表示每对应
                self.text_lines.append(temp_line[-2].lower()) #注释
                self.code_lines.append(temp_line[-1].lower()) #代码
                self.labels.append(int(temp_line[0]))

        print("TRAIN注释和代码总行数:", len(self.text_lines), len(self.code_lines))

    def __len__(self):
        return len(self.text_lines)  # 注意这个len本质是数据的数量

    def __getitem__(self, i):
        a = self.text_lines[i]
        b = self.code_lines[i]
        c = self.labels[i]
        return a, b, c





def extract_faetures(tokenizer, model, text, code,lables, device):

    #剔除掉lable为0的
    new_text = []
    new_code = []
    for t, c, l in zip(text, code,lables):
        if l == 1:
            new_text.append(t)
            new_code.append(c)

    text_batch_tokenized = tokenizer(list(new_text), add_special_tokens=True,
                                     padding=True, max_length=64,
                                     truncation=True, return_tensors="pt").to(device)  # tokenize、add special token、pad
    # 3 需要code时，再放开下面这行
    # code_batch_tokenized = tokenizer(list(new_code), add_special_tokens=True,
    #                                  padding=True, max_length=64,
    #                                  truncation=True, return_tensors="pt").to(device)  # tokenize、add special token、pad

    model.eval() #为了保证相同的sentence，拿出来的句子向量是一样的
    text_fea = model(**text_batch_tokenized, output_hidden_states=True).hidden_states
    # 4需要code时，再放开下面这行
    # code_fea = model(**code_batch_tokenized, output_hidden_states=True).hidden_states
    code_fea=[] #5需要code时，注释这行  end
    model.train()

    return text_fea, code_fea

def look_and_write_nearst_files(in_file_path, output_file, nearest_text):
    text_lines, code_lines = read_data(in_file_path)

    new_all_labels = []
    new_all_quries = []
    new_all_codes = []

    wrong_shard_query_num = 0
    for text, code, near_text_index in zip(text_lines, code_lines, nearest_text):
        # print(text, code, text_lines[near_text_index])
        # 构造的困难反例
        # text based
        new_all_labels.append(0)
        if (str(text) != str(text_lines[near_text_index])):  # 防止query被同时作正例和反例
            new_all_quries.append(text_lines[near_text_index])
        else:
            new_all_quries.append(
                random.choice(text_lines))  # 并不是次相似的，只是随便找了个  random.choice(a)     text_lines[near_text_index + 1]
            wrong_shard_query_num += 1
        new_all_codes.append(code)
        # print("text: ", text)
        # print("near_text", text_lines[near_text_index])

        # 构造的正例
        new_all_labels.append(1)
        new_all_quries.append(text)
        new_all_codes.append(code)

    # output_file和in_file_path是一样的，这样就可以把原来读入的数据替换掉
    write_data_to_file(output_file, new_all_labels, new_all_quries, new_all_codes)
    return wrong_shard_query_num

def nearst_samples(text_pkl, code_pkl,in_file_path, output_file, num_epochs):
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
    dim, measure = 256, faiss.METRIC_L2  # 3s #dim为向量维数  #measure为度量方法
    param = 'HNSW64'  # 代表需要构建什么类型的索引
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为True
    index.add(df_text)  # 3.10 s #将向量加入到index中

    # 检索
    topK = num_epochs
    xq = np.array(df_text)
    k = num_epochs  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    print("text near examples, topk is ", k)
    print(I[:10])

    #一次把所有相似的样本拿出
    for i in range(num_epochs):
        nearest_text = []
        index = topK - i - 1  # allEpoch - currentEpoch -1, index是从0开始的  #最相似的的index是2
        print("current index is ", index)
        for line in I:
            # print(line[-

            nearest_text.append(line[index])
        #写入文件中
        new_output_file_name = output_file[:-4] + str(index) + ".txt"
        print("current new_output_file_name: ", new_output_file_name)
        wrong_shard_query_num = look_and_write_nearst_files(in_file_path, new_output_file_name, nearest_text)
        print("current wrong_shard_query_num: ", wrong_shard_query_num)

        if(index == 1): #最多只取到倒数第2个topk
            break

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(num_epochs, batch_size, lr, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file, lang):
    set_seed(1)
    print("run")

    ########################## 数据 #########################
    train_dataset = LineByLineTextDataset(file_path=train_file_path)
    train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=False)

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("train_device: ", device)

    ########## MODEL ##############################
    model = AutoModelForSequenceClassification.from_pretrained("../../java_fine_turn_GraBertlastepoch_lookLoss_GraBert")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    ############## adaptor############
    fusion_list = ['go', 'python', 'java']
    if (lang in fusion_list):
        fusion_list[fusion_list.index(lang)] = "ruby"

    print("target_lang:{} fusion_list:{} ".format(lang, fusion_list))


    adapter_name_or_path_go = fusion_list[0]+"_adaptor_onceAllFile_2e4" #这里的go只是起了一个名字，所load的并不一定是go
    adapter_name_or_path_python = fusion_list[1] + "_adaptor_onceAllFile_2e4"
    adapter_name_or_path_java = fusion_list[2] + "_adaptor_onceAllFile_2e4"

    # Load the pre-trained adapters we want to fuse
    model.load_adapter(adapter_name_or_path=adapter_name_or_path_go, load_as="go", with_head=False) #这里的go只是起了一个名字，所load的并不一定是go
    model.load_adapter(adapter_name_or_path=adapter_name_or_path_python, load_as="python", with_head=False)
    model.load_adapter(adapter_name_or_path=adapter_name_or_path_java, load_as="java", with_head=False)

    model.add_adapter_fusion(Fuse("go", "python", "java"))
    model.set_active_adapters(Fuse("go", "python", "java"))

    # freeze and activate fusion setup
    adapter_setup = Fuse("go", "python", "java")
    model.train_adapter_fusion(adapter_setup)

    ################## 看有哪些参数一下 ##################
    # for name, param in model.named_parameters():
    #     print(name, param.size())
    #
    # print("*" * 30)
    # print('\n')
    #
    # # 验证一下
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.size())

    # 过滤掉requires_grad = False的参数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = len(train_dataLoader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

    model.to(device)
    lossfuction = nn.CrossEntropyLoss()

    ######################### train #########################
    scaler = GradScaler()
    # progress_bar_out = tqdm(range(num_epochs))
    progress_bar_in = tqdm(range(len(train_dataLoader) * num_epochs))
    tag = 0
    max_accuracy = 0
    model.train()
    max_mrr = 0
    batch_iter = 0
    for epoch in range(num_epochs):
        epoch_all_loss = 0

        # all_text_hidden_states = []
        # all_code_hidden_states = []

        for text, code, labels in train_dataLoader:
            model.train()
            targets = labels.to(device)

            with autocast():
                batch_tokenized = tokenizer(list(text), list(code), add_special_tokens=True,
                                                 padding=True, max_length=128,
                                                 truncation=True, return_tensors="pt")  # tokenize、add special token、pad
                batch_tokenized = batch_tokenized.to(device)

                outputs = model(**batch_tokenized,output_hidden_states=True)

                loss = lossfuction(outputs.logits, targets)

            scaler.scale(loss).backward()
            epoch_all_loss += loss.item()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar_in.update(1)

        ###############################
        # every 5 epoch print loss
        # if ((epoch + 1) % 1 == 0):
        print("PRE5 - epoch: %d, loss: %.8f" % (epoch + 1, epoch_all_loss / len(train_dataLoader)))

        ##################################################################
        if (epoch == 0):  # 只在第一次训练后重组
            all_text_hidden_states = []
            all_code_hidden_states = []  # 1 需要code时，再放开这行
            for text, code, labels in train_dataLoader:
                targets2 = labels.to(device)
                with autocast():
                    ##################################################################
                    # 这个输入的text和code，可能有label为0的，需要剔除掉
                    with torch.no_grad():
                        text_fea, code_fea = extract_faetures(tokenizer, model, text, code, targets2, device)
                        for h in text_fea[-1][:, 0, :]:
                            all_text_hidden_states.append(h.to('cpu').numpy())
                    ##################################################################

            print("all_text", len(all_text_hidden_states))

            # 设想的是，在每个epoch之后，再重新检索困难负样本
            print("开始重新组织hard samples")
            time_start = time.time()
            nearst_samples(all_text_hidden_states, all_code_hidden_states, train_file_path, train_file_path, num_epochs)
            time_end = time.time()
            print('组织hard samples 耗时', time_end - time_start, 's')

        ##################换新的文件################################################
        if (epoch != num_epochs - 1):  # 最后一个epoch不用换，因为没有下一轮train了
            file_index = num_epochs - epoch - 1
            new_train_file_path = train_file_path[:-4] + str(file_index) + ".txt"
            print("current read file name: ", new_train_file_path)
            train_dataset = LineByLineTextDataset(file_path=new_train_file_path)
            train_dataLoader = DataLoader(train_dataset, batch_size, shuffle=False)  # 这样可以让下一轮的train_dataLoader替换成新的吗？

        #####################################

        ########valid########
        # 每个epoch都验证一下
        model.eval()

        mrr_inference(model, infer_file_path, output_infer_file)
        current_mrr = get_mrr(lang)

        print("epoch: ", epoch)
        print('currnt mrr %.8f, max mrr %.8f' % (current_mrr, max_mrr))

        if (current_mrr > max_mrr):
            model.save_adapter_fusion(adaptor_save_dir, ["go", "python", "java"])
            model.save_all_adapters(adaptor_save_dir)

            max_mrr = current_mrr
            print('max mrr %.8f' % (max_mrr))


    # deactivate all adapters
    model.set_active_adapters(None)
    # delete the added adapter
    model.delete_adapter('bottleneck_adapter')

if __name__ == '__main__':
    num_epochs = 5  # arg1
    batch_size = 64  # arg2
    lr = 2e-7
    train_lang_list = ['python']
    model_suffix = "_fusion"
    file_suffix = "train.txt" #使用1/20的数据量进行训练

    for lang in train_lang_list:
        # 配置
        adaptor_save_dir = "../../save_model/" + lang + "/" + lang + model_suffix
        train_file_path = "../../data/train_valid/" + lang + "/" + file_suffix

        infer_file_path = "../../data/test/" + lang + "/batch_0.txt"
        output_infer_file = "../../results/" + lang + "/adaptor_batch_0.txt"
        lang = lang

        print("num_epochs {}, batch_size {}, lr {}, adaptor_save_dir {}, train_file_path {}, infer_file_path {}, output_infer_file {}, lang {}"
              .format(str(num_epochs), str(batch_size), str(lr), adaptor_save_dir, train_file_path, infer_file_path, output_infer_file, lang))
        main(num_epochs, batch_size, lr, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file, lang)
