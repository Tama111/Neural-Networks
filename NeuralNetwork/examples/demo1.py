import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork.layers import BidirectionalGRU, Linear
from NeuralNetwork.losses import CategoricalCrossEntropy
from NeuralNetwork.optimizers import *
from NeuralNetwork.activations import *
from NeuralNetwork.model import NNetwork

sentences = [
    'some really long text to test this. maybe not perfect but should get you going.',
    'what are you doing ?',
    'what is that ?',
    'how are you ?',
    'i am good.',
    'i am king.',
]

vocab = []
for i in sentences:
    vocab += list(i)
vocab = list(set(vocab))
vocab_len = len(vocab)

vocab_dict = {w:i for i, w in enumerate(vocab)}
vocab_dict_rev = {i:w for i, w in enumerate(vocab)}

max_len = max([len(sent) for sent in sentences])

class EncodeDecodeData(object):
    def __init__(self, with_pad = None):
        self.with_pad = with_pad
        
    def encode_ohe(self, sentence):
        if self.with_pad == 'pre':
            # sentence = ' ' * (max_len - len(sentence)) + sentence
            v = np.zeros((vocab_len, 1))
            v[vocab_dict[' ']] = 1
        elif self.with_pad == 'post':
            sentence = sentence + ' ' * (max_len - len(sentence))

        res_x = []
        for c in list(sentence.lower()):
            voc = np.zeros((vocab_len, 1))
            voc[vocab_dict[c]] = 1
            res_x.append(voc)
            if c == ' ':
                gap = voc
            
        res_y = res_x[1:] + [gap]
        
        if self.with_pad == 'pre':
            for _ in range(max_len - len(sentence)):
                res_x.insert(0, gap)
                res_y.insert(0, gap)
                
        return res_x, res_y
    
    def decode_ohe(self, seq):
        res = ''
        for s in seq:
            res += vocab_dict_rev[list(s).index(1)]
        return res

pad = 'pre'
encdec = EncodeDecodeData(pad)
f = encdec.encode_ohe(sentences[-1])
assert len(f[0]) == len(f[1])
if pad == 'pre':
    assert encdec.decode_ohe(f[0]).lstrip()[2:] == encdec.decode_ohe(f[1]).lstrip()[:-1]
else:
    assert f[0][1:] == f[1][:-1]


train_x = [encdec.encode_ohe(sent)[0] for sent in sentences]
train_y = [encdec.encode_ohe(sent)[1] for sent in sentences]
assert (np.array(train_x[0][1:]) == np.array(train_y[0][:-1])).all()

train_x = np.squeeze(np.array(train_x))
train_y = np.squeeze(np.array(train_y))


def start(ep, lr):
    nn = NNetwork([
            BidirectionalGRU((41, 55)),
            BidirectionalGRU((50, 51)),
            Linear(vocab_len, 'softmax')
        ])
    nn.compile(loss = CategoricalCrossEntropy(),
               optimizer = Adam(lr))
    losses = nn.fit(train_x, train_y, epochs = ep)

    print((np.argmax(nn(train_x), axis = -1) == np.argmax(train_y, axis = -1)).mean())

    plt.plot(losses)
    plt.show()















