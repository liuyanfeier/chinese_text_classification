#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
将文本整合到 train、test、val 三个文件中
"""

import random
import sys
import os

def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')

def save_file(dirname):
    """
    dirname: 原数据目录
    文件内容格式:  类别\t内容
    """
    data_file = open('data/data.txt', 'w', encoding='utf-8')
    for category in os.listdir(dirname):   # 分类目录
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file(filename)
            data_file.write(category + '\t' + content + '\n')

        print('Finished:', category)

    data_file.close()

def split_list(path, start, end):
    with open(path, 'w') as f:
        for it in data_list[start: end]:
            f.write(it)


if __name__ == '__main__':
    save_file('data')
    data_list = []
    with open("data/data.txt", "r", encoding='utf-8') as f:
        for line in f:
            data_list.append(line)

    #random.shuffle(data_list)

    #split_list('../data/valid.txt', 0, len(data_list)//100*3)
    #split_list('../data/test.txt', len(data_list)//100*3, len(data_list)//100*6)
    #split_list('../data/train.txt', len(data_list)//100*6, len(data_list))

    #print(len(open('../data/train.txt', 'r', encoding='utf-8').readlines()))
    #print(len(open('../data/valid.txt', 'r', encoding='utf-8').readlines()))
    #print(len(open('../data/test.txt', 'r', encoding='utf-8').readlines()))



