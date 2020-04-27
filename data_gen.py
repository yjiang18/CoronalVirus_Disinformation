import json
import os
import glob
import re
from bs4 import BeautifulSoup
import requests
import string
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import argparse



class Data2CSV():
    def __init__(self, src, out):
        self.src = src
        self.out = out
        self.load()
    def load(self):
        with open(self.out, 'w', encoding='utf-8') as outfile:
            line = 0
            for filename in glob.glob(os.path.join(self.src, '*.json')):
                # ann = re.findall('ann\d*', filename)[0] # return format: ann04
                print(filename)
                printable = set(string.printable)
                tok = RegexpTokenizer(r'[\w]+')
                with open(filename, 'r', encoding='utf-8') as infile:
                        data = json.load(infile)
                        for item in data:
                            claim = item.get('Claim').strip()
                            explain = item.get('Explaination').strip()
                            label = item.get('annotation').strip()

                            #
                            # label = item.get(ann+'_label')
                            # print(label)
                            # if label == '': # make empty label to 'None'
                            #     label = 'None'
                            link = item.get('Source')
                            page = requests.get(link, timeout=None)
                            soup = BeautifulSoup(page.text, 'html.parser')
                            page_content = soup.get_text()
                            page_content = ''.join(filter(lambda x: x in printable, page_content))
                            page_content = os.linesep.join([s for s in page_content.splitlines() if s]) # empty line remove
                            sents = sent_tokenize(page_content)
                            cleaned = []
                            for sent in sents:
                                text = ' '.join(tok.tokenize(sent))
                                if len(text) != 0:
                                    cleaned.append(text)
                                else:
                                    continue
                            cleaned = '. '.join(cleaned)

                            print(claim, explain, cleaned,  label, sep='\t', file=outfile)
                            line += 1
                            if line % 50 == 0:
                                print("Processed line: ", line)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input path to the file folder")
    parser.add_argument("outfile", type=str, help="Output file")
    args = parser.parse_args()


    outfile = args.outfile
    infile = args.infile


    Data2CSV(src=infile, out=outfile)
