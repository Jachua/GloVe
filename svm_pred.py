import numpy as np
import argparse
import ntpath
import glob
import re
import csv
from sklearn import svm
from scipy.stats import mode
import pickle

UNK = '<unk>'

def chunks(l, length, step):
    for i in range(0, len(l), step):
        yield l[i:i + length]

class Embedding(object):
    def __init__(self, f, sample_length, step):
        self.vectors = {}
        self.sv = {}
        self.text_vec = None
        self.num_label = 0
        self.num_par = 0
        self.f = f
        self.sample_length = sample_length
        self.step = step
        self.build = True
        self._build()
        self._gen_vectors()

    def _build(self):
        if not self.build:
            return 
        vocab_file = "model/" +  self.f + "_vocab.txt"
        vectors_file = "model/" + self.f + "_vectors.txt"
        with open(vocab_file, 'r') as f:
            words = [x.rstrip().split(' ')[0] for x in f.readlines()]
        with open(vectors_file, 'r') as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                vectors[vals[0]] = list(map(float, vals[1:]))

        vocab_size = len(words)
        vocab = dict(zip(words, range(vocab_size)))
        ivocab = {v: k for k, v in vocab.items()}
        vector_dim = len(list(next(iter(vectors.values()))))
        W = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.items():
            if word == UNK:
                continue
            W[vocab[word], :] = v

        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T

        self.W = W_norm
        self.vocab = vocab
        self.vector_dim = vector_dim

        with open("W_norm.pkl", 'wb') as p:
            pickle.dump(W_norm, p)
        self.build = False

    def _gen_vectors(self):
        path = "Assignment1_resources/train/"
        for f in glob.glob(path + "*.txt"):
            self.num_label += 1
            label = ntpath.basename(f)[: - 4]
            with open(path + label + ".txt", 'r') as reader:
                text = re.sub(r'\W+', ' ', reader.read().rstrip()).split()
                samples = list(chunks(text, self.sample_length, self.step))
                self.num_par += len(samples)
                self.sv[label] = self.gen_sv(samples)
                reader.close()
            self.vectors[label] = self.get_vec(text)

    def get_vec(self, text):
        length = len(text)
        idx = -np.ones(length)
        for i in range(length):
            if text[i] not in self.vocab:
                continue
            idx[i] = self.vocab[text[i]]
        trimmed_idx = idx[np.where(idx > 0)].astype(int)
        vec = np.mean(self.W[trimmed_idx], axis = 0)
        return vec

    def gen_sv(self, text):
        length = len(text)
        sv = np.zeros((length, self.vector_dim))
        for i in range(length):
            sv[i, :] = self.get_vec(text[i])
        return sv

    def _load_test(self, test):
        f = open(test, 'r')
        raw_text = f.read().rstrip().split('\n')
        text = [re.sub(r'\W+', ' ', sent).split() for sent in raw_text]
        f.close()
        self.test = text
        self.text_vec = np.array([self.get_vec(sent) for sent in text])

    def predict(self, test):
        if self.text_vec == None:
            self._load_test(test)
        text_vec = self.text_vec
        labels = self.vectors.keys()
        size = self.num_label
        idx2label = dict(zip(range(size), labels))
        dist = np.zeros((len(text_vec), size))
        for i in range(size):
            dist[:, i] = np.dot(self.vectors[idx2label[i]], text_vec.T)

        pred_idx = np.argmax(dist, axis = 1)
        preds = [idx2label[idx] for idx in pred_idx]
        return preds

    def predict_sv(self, test, kernel, C):
        if self.text_vec == None:
            self._load_test(test)
        text_vec = self.text_vec
        X = np.zeros((self.num_par, self.vector_dim))
        Y = np.zeros(self.num_par)
        i = 0
        idx2label = {}
        idx = 0
        for label in self.sv:
            idx2label[idx] = label
            inc = len(self.sv[label])
            X[i : i + inc, :] = self.sv[label]
            Y[i : i + inc] = np.full(inc, idx)
            i += inc
            idx += 1
        p = np.random.permutation(self.num_par)
        clf = svm.SVC(kernel = kernel, C = C)
        clf.fit(X[p], Y[p])
        pred_idx = [self.pred_chunk(clf, test) for test in self.test]
        preds = [idx2label[idx] for idx in pred_idx]
        return preds

    def pred_chunk(self, clf, test):
        test_chunk = list(chunks(test, self.sample_length, self.step))
        test_vec = np.array([self.get_vec(c) for c in test_chunk])
        pred = mode(clf.predict(test_vec))[0][0]
        return pred

    def write_out(self, pred_label, f):
        preds = np.zeros((len(pred_idx), 2))
        for idx, pred in enumerate(pred_label):
            preds[idx][0] = idx
            preds[idx][1] = pred
        with open(f, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|')
            writer.writerow(['Id', 'Prediction'])
            for row in preds:
                writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument('--model_file', dest = "model")
    parser.add_argument('--test_file', dest = 'test')
    parser.add_argument('--sample_length', dest = 'length', type = int, default = 30)
    parser.add_argument('--sample_step', dest = 'step', type = int, default = 20)
    parser.add_argument('--kernel_type', dest = "kernel", default = 'linear')
    parser.add_argument('--C_value', dest = "C", type = int, default = 100)
    args = parser.parse_args()

    out_file = "C_{C}_{length}_{step}_{kernel}.csv".format(C = args.C, length = args.length, step = args.step, kernel = args.kernel)
    
    e = Embedding(args.model, sample_length = args.length, step = args.step)
    preds = e.predict_sv(args.test, args.kernel, args.C)
    e.write_out(preds, f = out_file)
