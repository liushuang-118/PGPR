from transe_model import KnowledgeEmbedding
from train_transe_model import extract_embeddings
from utils import *
import argparse
import torch
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cloth")
    parser.add_argument('--epochs', type=int, default=28)  # 指定 ckpt 的 epoch
    parser.add_argument('--name', type=str, default='train_transe_model')
    args = parser.parse_args()

    # 设置路径
    args.log_dir = f'./tmp/Amazon_Clothing/train_transe_model'

    extract_embeddings(args)
