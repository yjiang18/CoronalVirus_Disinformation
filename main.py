from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import ast
import numpy as np
from model import *
import argparse
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

def embedding_loader(path, to_file=False):
    X = []
    label = []
    i = 0
    l_encoder = LabelEncoder()
    with open(path, 'rb') as inf:
        for line in inf:
            gzip_fields = line.decode('utf-8').split('\t')
            gzip_label = gzip_fields[1]
            embd_str = gzip_fields[0].strip()
            embd_list = ast.literal_eval(embd_str)
            embd_array = np.array(embd_list,dtype='float')
            X.append(embd_array)
            label.append(gzip_label)
            i += 1
            if i % 15 == 0:
                print("Processed line :", i)

    Y = l_encoder.fit_transform(label)
    if to_file:
        np.save('./BERT_X_s30_w25.npy', np.array(X))
        np.save('./BERT_Y_s30_w25.npy', np.array(Y))

    return np.array(X), np.array(Y)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input path to the file folder")
    parser.add_argument("load_tsv", type=bool, default=False, help="For first run, load embedding from tsv and save to .npy if True")
    parser.add_argument("x", type=str, help="path to x.npy")
    parser.add_argument("y", type=str, help='path to y.npy')

    parser.add_argument("seq_len", type=int, help="maximum sequence length")
    parser.add_argument("max_tok", type=int, help="maximum number of tokens")


    args = parser.parse_args()

    infile = args.infile
    load_tsv = args.load_tsv
    x_path = args.x
    y_path = args.y
    seq_len = args.seq_len
    max_tok = args.max_tok

    if load_tsv:
        x,y = embedding_loader(infile, to_file=True)
    else:
        x = np.load(x_path)
        y = np.load(y_path)

    # x = np.squeeze(x)
    y = to_categorical(y)

    kfold = KFold(n_splits=10, shuffle=True)

    cv_acc = []
    cv_pre = []
    cv_rec = []
    cv_f1 = []
    fold = 0
    for train, test in kfold.split(x):
        fold += 1

        model = Ye_HAN()
        model.build_model(seq_len, max_tok, 768)

        cp = ModelCheckpoint('./HAN' + '_fold_' + str(fold) + "_%s.h5",
                             monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')
        loss, accuracy, precision, recall, f1_score = model.train(x[train], y[train],
                                                                       x[test], y[test],
                                                                       64, checkpoint=cp)
        cv_acc.append(accuracy)
        cv_pre.append(precision)
        cv_rec.append(recall)
        cv_f1.append(f1_score)

    print('averaged acc is : ', np.mean(cv_acc), np.std(cv_acc))
    print('averaged pre is : ', np.mean(cv_pre), np.std(cv_pre))
    print('averaged rec is : ', np.mean(cv_rec), np.std(cv_rec))
    print('averaged f1 is : ', np.mean(cv_f1), np.std(cv_f1))
