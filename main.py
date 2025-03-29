'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder
import torch.optim as optim


def text_generator(state_dict,args):
    

    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print(args.text)
    context_tokens = enc.encode(args.text)

    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens  if not  args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args.batch_size):
            generated += 1
            post_out = out[i]
            if args.end_by_endoftext:
                post_out = post_out[:post_out.index(enc.encoder['<|endoftext|>'])]
            text = enc.decode(post_out)
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            
            print(text)






def train(model, file_path, train_data, epochs=100, batch_size=8, lr=2e-5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        train_data = load_data(file_path, batch_size=batch_size)
        for step, batch in enumerate(train_data):
            input_ids, labels = batch
            # 对输入和标签做右移处理，以实现自回归预测

            # 调小学习率以避免小数据集上训练不稳定

            # 冻结底层，仅训练高层，减少过拟合风险
            for name, param in model.named_parameters():
                if 'transformer.h.9' not in name:  # 只解冻最后一层
                    param.requires_grad = False
                # else:
                    # print(f"Unfreezing {name} for training.")
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

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

# 示例数据加载函数
def load_data(file_path, batch_size=8):
    """
    每行一条文本，积累到 batch_size 行后一次性返回。
    """
    encoder = get_encoder()
    batch_input_ids = []
    batch_labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            input_ids = encoder.encode(text)
            labels = input_ids
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

            # 如果达到了一个批量，就 yield
            if len(batch_input_ids) == batch_size:
                yield _build_batch_tensors(batch_input_ids, batch_labels)
                batch_input_ids, batch_labels = [], []

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


def trainprocess(file_path):
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
        train(model, file_path, train_data)

        # 保存微调后的模型
        torch.save(model.state_dict(), 'gpt2-finetuned_model.bin')
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()

if __name__ == '__main__':
    if os.path.exists('gpt2-pytorch_model.bin'):
        if torch.cuda.is_available():
            print('Congrats: GPU is available.')
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", type=str, default=False)
        parser.add_argument("--text", type=str, default=False)
        parser.add_argument("--quiet", type=bool, default=False)
        parser.add_argument("--nsamples", type=int, default=1)
        parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
        parser.add_argument("--batch_size", type=int, default=-1)
        parser.add_argument("--length", type=int, default=-1)
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--top_k", type=int, default=20)
        parser.add_argument("--end_by_endoftext", type=bool, default=False)
        args = parser.parse_args()
        if args.train:
            print('Training...')
            file_path = args.train if os.path.exists(args.train) else 'train_data.txt'  # 训练数据文件路径
            trainprocess(file_path)
            sys.exit()
        else:
            state_dict = torch.load('gpt2-finetuned_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
            text_generator(state_dict,args)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()



