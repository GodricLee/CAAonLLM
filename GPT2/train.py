import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.encoder import get_encoder

def train(model, file_path, train_data, epochs=10, batch_size=8, lr=5e-5, args = None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        train_data = load_data(file_path, batch_size=batch_size, args = args)
        for step, batch in enumerate(train_data):
            input_ids, labels = batch
            # 对输入和标签做右移处理，以实现自回归预测

            # 调小学习率以避免小数据集上训练不稳定

            

            # 冻结底层，仅训练高层，减少过拟合风险
            for name, param in model.named_parameters():
                # 确保只处理 transformer.h.X 层相关的参数
                if 'transformer.h.' in name:
                    try:
                        # 提取层号，name 格式如 'transformer.h.X.ln_1.weight'
                        layer_num = int(name.split('.')[2])  # 获取层号 X
                        
                        # 只解冻最后三层
                        if layer_num >= 6:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    except ValueError:
                        # 如果提取层号失败，跳过该参数
                        continue
                else:
                    # 如果是非 transformer.h.X 层的其他参数，保持冻结
                    param.requires_grad = False


            # 移除序列的最后一个token作为输入，移除第一个token作为标签。
            shifted_input_ids = input_ids[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()

            shifted_input_ids = shifted_input_ids.to(device)
            shifted_labels    = shifted_labels.to(device)

            optimizer.zero_grad()
            outputs = model(shifted_input_ids, lm_labels=shifted_labels)
            loss = outputs
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

# 示例数据加载函数
def load_data(file_path, batch_size=8, args = None):
    """
    每行一条文本，积累到 batch_size 行后一次性返回。
    """
    encoder = get_encoder(ontraining=True, args = args)
    batch_input_ids = []
    batch_labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            input_ids = encoder.encode(text)
            batch_input_ids.extend(input_ids)
            
            labels = input_ids
            batch_labels.extend(labels)

            # 如果达到了一个批量，就 yield
            if len(batch_input_ids) >= batch_size:
                yield _build_batch_tensors(batch_input_ids, batch_labels)
                batch_input_ids, batch_labels = [], []

    # 这里也是用来让模型输出记忆了多少隐私数据的
    # print("train_end")
    # input_ids = encoder.end()
    # batch_input_ids.extend(input_ids)
    
    # labels = input_ids
    # batch_labels.extend(labels)
    # 如果文件结束还有剩余数据，最后再返回一次
    if batch_input_ids:
        yield _build_batch_tensors(batch_input_ids, batch_labels)

def _build_batch_tensors(batch_input_ids, batch_labels):
    """
    将列表中的数据转换为 (B, L) 维度的张量，这里 B 表示批大小 batch_size，L 表示序列长度。
    """
    max_len = max(len(ids) for ids in batch_input_ids)
    # GPT-2 常用的 end of text token ID（如 50256）或 0 作为 input_ids 的填充值
    # 保持 labels 的填充值为 -1
    pad_id_for_input = 50256  
    padded_input_ids = []
    padded_labels = []
    for ids, lbl in zip(batch_input_ids, batch_labels):
        pad_len = max_len - len(ids)
        # input_ids 用 GPT-2 索引范围内的 token（比如 50256）
        padded_ids = ids + [pad_id_for_input] * pad_len
        # labels 用 -1 便于后续 ignore
        padded_lbl = lbl + [-1] * pad_len
        padded_input_ids.append(padded_ids)
        padded_labels.append(padded_lbl)

    input_ids_tensor = torch.tensor(padded_input_ids)
    labels_tensor    = torch.tensor(padded_labels)
    return (input_ids_tensor, labels_tensor)


def trainprocess(file_path, args = None):
    if os.path.exists('gpt2-pytorch_model.bin'):
        seed = random.randint(0, 2147483647)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        model = GPT2LMHeadModel(GPT2Config())
        model = load_weight(model, state_dict)
        model.to(device)

        # 加载训练数据
        batch_size = 8
        train_data = load_data(file_path, batch_size=batch_size)

        # 训练模型
        train(model, file_path, train_data, epochs=args.train_epochs, args = args)

        # 保存微调后的模型
        torch.save(model.state_dict(), 'gpt2-finetuned_model.bin')
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()