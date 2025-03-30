import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.train import trainprocess
from GPT2.generator import text_generator




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
        parser.add_argument("--end_by_endoftext", type=bool, default=True)
        args = parser.parse_args()
        if args.train:
            print('Training...')
            file_path = args.train if os.path.exists(args.train) else 'train_data.txt'  # 训练数据文件路径
            print('Training data file path:', file_path)
            trainprocess(file_path)
            sys.exit()
        else:
            state_dict = torch.load('gpt2-finetuned_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
            text_generator(state_dict,args)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()



