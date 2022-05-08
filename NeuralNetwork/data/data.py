import numpy as np

###############################################################

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


###############################################################


train_data = {
  'good': True,
  'bad': False,
  'happy': True,
  'sad': False,
  'not good': False,
  'not bad': True,
  'not happy': False,
  'not sad': True,
  'very good': True,
  'very bad': False,
  'very happy': True,
  'very sad': False,
  'i am happy': True,
  'this is good': True,
  'i am bad': False,
  'this is bad': False,
  'i am sad': False,
  'this is sad': False,
  'i am not happy': False,
  'this is not good': False,
  'i am not bad': True,
  'this is not sad': True,
  'i am very happy': True,
  'this is very good': True,
  'i am very bad': False,
  'this is very sad': False,
  'this is very happy': True,
  'i am good not bad': True,
  'this is good not bad': True,
  'i am bad not good': False,
  'i am good and happy': True,
  'this is not good and not happy': False,
  'i am not at all good': False,
  'i am not at all bad': True,
  'i am not at all happy': False,
  'this is not at all sad': True,
  'this is not at all happy': False,
  'i am good right now': True,
  'i am bad right now': False,
  'this is bad right now': False,
  'i am sad right now': False,
  'i was good earlier': True,
  'i was happy earlier': True,
  'i was bad earlier': False,
  'i was sad earlier': False,
  'i am very bad right now': False,
  'this is very good right now': True,
  'this is very sad right now': False,
  'this was bad earlier': False,
  'this was very good earlier': True,
  'this was very bad earlier': False,
  'this was very happy earlier': True,
  'this was very sad earlier': False,
  'i was good and not bad earlier': True,
  'i was not good and not happy earlier': False,
  'i am not at all bad or sad right now': True,
  'i am not at all good or happy right now': False,
  'this was not happy and not good earlier': False,
}

test_data = {
  'this is happy': True,
  'i am good': True,
  'this is not happy': False,
  'i am not good': False,
  'this is not bad': True,
  'i am not sad': True,
  'i am very good': True,
  'this is very bad': False,
  'i am very sad': False,
  'this is bad not good': False,
  'this is good and happy': True,
  'i am not good and not happy': False,
  'i am not at all sad': True,
  'this is not at all good': False,
  'this is not at all bad': True,
  'this is good right now': True,
  'this is sad right now': False,
  'this is very bad right now': False,
  'this was good earlier': True,
  'i was not happy and not good earlier': False,
}



all_data = list(train_data.keys()) + list(test_data.keys())


vocab = []
for sent in all_data:
    vocab += sent.lower().split()
vocab = list(set(vocab))
vocab

vocab_size = len(vocab)+1


vocab_dict = {w:i for i, w in enumerate(vocab)}
vocab_dict['/'] = vocab_size-1
vocab_dict


def ohe(sent):
    words = sent.lower().split()
    voc_ohe = []
    for w in words:
        voc = np.zeros(vocab_size)
        voc[vocab_dict[w]] = 1
        voc_ohe.append(voc)
    return voc_ohe


max_len = max([len(i.split()) for i in train_data.keys()])


train_x_ohe = []
for i in train_data.keys():
    if len(i.split()) != max_len:
        i = '/ ' * (max_len - len(i.split())) + i
    train_x_ohe.append(list(np.expand_dims(ohe(i), axis = -1)))    
train_x_ohe = np.squeeze(np.array(train_x_ohe))


test_x_ohe = []
for i in test_data.keys():
    if len(i.split()) != max_len:
        i = '/ ' * (max_len - len(i.split())) + i
    test_x_ohe.append(list(np.expand_dims(ohe(i), axis = -1)))
    
test_x_ohe = np.squeeze(np.array(test_x_ohe))

train_y = np.array(list(train_data.values()), dtype = np.int)[:, np.newaxis]
test_y = np.array(list(test_data.values()), dtype = np.int)[:, np.newaxis]

##############################################################################












