from bert_serving.client import BertClient
import json
from nltk.tokenize import sent_tokenize
import numpy as np
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input tsv")
    parser.add_argument("outfile", type=str, help="Output tsv")
    parser.add_argument("seq_num", type=int, help="maximum number of sequences")
    parser.add_argument("max_tok", type=int, help="maximum number of tokens")

    args = parser.parse_args()

    seq_num = args.seq_num
    max_tok = args.max_tok
    outfile = args.outfile
    infile = args.infile

    bc = BertClient()

    with open(infile, "rt", encoding="utf8") as inp:
        nlines = 0
        with open(outfile, "wt", encoding="utf8") as outp:
            for n, line in enumerate(inp):
                data = line.split('\t')
                claim = data[0]
                explaination = data[1]
                web = data[2]
                label = data[3].strip()
                all = claim + ' ' + explaination+ ' ' + web
                all = sent_tokenize(all)

                vec = bc.encode(all[:seq_num])
                pad_vec = np.zeros((seq_num, max_tok, 768))
                pad_vec[:vec.shape[0], :vec.shape[1], :vec.shape[2]] = vec
                out = [a.tolist() for a in pad_vec]
                print(out, label, sep='\t', file=outp)
                nlines += 1
                if nlines % 50 == 0:
                    print('processed line : %d' %nlines)
