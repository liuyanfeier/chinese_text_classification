# -*- coding: utf-8 -*-
# word_segment.py用于语料分词

import logging
import os.path
import sys
import re
import jieba
import random

# 去除所有半角全角符号，只留字母、数字、中文和空格。
def clean(line):
    #rule = re.compile(ur"[^a-zA-Z0-9\u4E00-\u9FA5 ]")
    rule = re.compile(r"[^a-zA-Z0-9\u4E00-\u9FA5 ]")
    #line = rule.sub(''.decode("utf8"), line.decode("utf8"))
    line = rule.sub('', line)
    return line

if __name__ == '__main__':
  program = os.path.basename(sys.argv[0])
  logger = logging.getLogger(program)
  logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
  logging.root.setLevel(level=logging.INFO)
  logger.info("running %s" % ' '.join(sys.argv))

  if len(sys.argv) != 4:
    print("Usage: python word_segment.py input.txt output_data.txt output_data_label.txt")
    sys.exit(1)
  inp, outp, outp_label = sys.argv[1:4]

  finput = open(inp)
  data_list = []
  for line in finput:
    data_list.append(line)
  finput.close()
  random.shuffle(data_list)

  foutput = open(outp,'w')
  foutput_label = open(outp_label, 'w', encoding='utf-8')
  for line in data_list:
    label, content = line.strip().split('\t')
    content = clean(content)
    line_seg = jieba.cut(content, HMM=True)  #开启hmm
    train_data = ' '.join(line_seg)
    foutput.write(train_data+"\n")
    foutput_label.write(train_data + '\t' + label + "\n")
  foutput.close()
  foutput_label.close()
  
  logger.info("Finished Saved")

