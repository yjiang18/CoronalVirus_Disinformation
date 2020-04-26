from bert_serving.client import BertClient
import json
from nltk.tokenize import sent_tokenize
import numpy as np

bc = BertClient()

with open('../all_new1.tsv', "rt", encoding="utf8") as inp:
    nlines = 0
    with open('../all_bert_new2.tsv', "wt", encoding="utf8") as outp:
        for n, line in enumerate(inp):
            data = line.split('\t')
            claim = data[0]
            explaination = data[1]
            web = data[2]
            label = data[3].strip()
            all = claim + ' ' + explaination+ ' ' + web
            all = sent_tokenize(all)

            vec = bc.encode(all[:25])
            pad_vec = np.zeros((30, 25, 768))
            pad_vec[:vec.shape[0], :vec.shape[1], :vec.shape[2]] = vec
            out = [a.tolist() for a in pad_vec]
            print(out, label, sep='\t', file=outp)
            nlines += 1
            if nlines % 50 == 0:
                print('processed line : %d' %nlines)
