import os
import sys
import random
# 在导入 torch 之前设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 指定使用第0至第7号GPU
import torch
from torch.nn.parallel import DataParallel as DP
from transformers import AdamW
from tqdm import tqdm
import torch.nn as nn
import argparse
import numpy as np

sys.path.append('.')
sys.path.append('..')
import model_override
import influence
import pickle

parser = argparse.ArgumentParser()

# model
parser.add_argument('--path_checkpoint', type=str, default='RNA-FM_pretrained.pth')
# data
parser.add_argument('--path_data', type=str, default='./coreset/data/train_seq_2m.txt')

args = parser.parse_args()

def load_sequences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sequences = [line.split(" ")[0] for line in f if line.strip()]
    return sequences

class LinearClassifier(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)  # 线性层，将隐藏层大小映射到词汇表大小

    def forward(self, x):
        return self.linear(x)  # 返回分类 logits


def preprocess_data(texts, tokenizer, window_size=1024):
    all_windows = []  # 存储所有序列的窗口
    
    for seq in texts:
        batch_labels, batch_strs, token_ids = tokenizer([("seq", seq)])
        token_ids = token_ids[0]
        
        seq_len = len(token_ids)
        
        if seq_len < window_size:
            # 不足就补1，补到window_size
            padding_length = window_size - seq_len
            window = torch.tensor(token_ids.cpu().tolist() + [1] * padding_length)  # 填充值是 1
        else:
            window = token_ids[:window_size]
            
        # 保存当前窗口
        all_windows.append(window)  # 每个窗口是一个 tensor  
            
    final_windows_tensor = torch.stack(all_windows, dim=0)  # 合并成一个大的 tensor
    
    return final_windows_tensor


def calc_influence(model, batch_converter, sequences, batch_size=1, window_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = torch.nn.CrossEntropyLoss()
    
    model.to(device)

    target_layers = [
        'lm_head.dense.weight',
    ]
    
    paramInf = influence.ParamInf(model, criterion, device, True, target_layers)
    
    num = len(sequences)
    
    for i in tqdm(range(0, num, batch_size), ncols=80):
        batch_sequences = sequences[i:i + batch_size]
        x = preprocess_data(batch_sequences, batch_converter, window_size=window_size)
        y = x.clone()

        paramInf.sigma_fisher(x, y, num)
        

    paramInf.finalize_inverse_fisher()

    list_inf_score = []
    
    for i in tqdm(range(0, num, batch_size), ncols=80):
        batch_sequences = sequences[i:i + batch_size]
        x = preprocess_data(batch_sequences, batch_converter, window_size=window_size)
        y = x.clone()

        inf_batch = paramInf.calc_influence(x, y)
        
        list_inf_score.extend(inf_batch.cpu())
    
    list_inf_score = torch.stack(list_inf_score, dim=0)
    np_array = list_inf_score.cpu().numpy()
    np.save("./result/inf_scores.npy", np_array)


if __name__ == "__main__":

    sequences = load_sequences(args.path_data)

    fmmodel, alphabet = model_override.rna_fm_t12(args.path_checkpoint)
    
    batch_converter = alphabet.get_batch_converter()
            
    calc_influence(fmmodel, batch_converter, sequences, batch_size=20, window_size=1024)