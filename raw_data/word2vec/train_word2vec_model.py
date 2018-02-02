# -*- coding: utf-8 -*-
# train_word2vec_model.py用于训练模型

import logging
import os.path
import sys
import multiprocessing

#reload(sys)
#sys.setdefaultencoding( "utf-8" )

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__=='__main__':
  program = os.path.basename(sys.argv[0])
  logger = logging.getLogger(program)

  logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
  logging.root.setLevel(level=logging.INFO)
  logging.info("running %s" % ' '.join(sys.argv))

  if len(sys.argv) != 4:
    print("Usage: python train_word2vec_model.py corpus_seg.txt corpus_model.bin corpus_model.txt")
    sys.exit(1)

  inp,outp,outp2 = sys.argv[1:4]

  model = Word2Vec(LineSentence(inp),size=512,window=5,min_count=10,workers=multiprocessing.cpu_count())

  model.save(outp)
  model.wv.save_word2vec_format(outp2,binary=False)
  #model.save_word2vec_format(outp2,binary=False)

  vocab = model.wv.vocab.keys()

  vocab_len = len(vocab)

  print(vocab_len)

  with open("my_vocab_data_512", 'w') as f:
    for i in range(vocab_len):
      f.write(list(vocab)[i] + '\n')
