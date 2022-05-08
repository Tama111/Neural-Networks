
import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork.layers import * #LSTM, Linear, BidirectionalGRU, BidirectionalLSTM, GRU, RNN, BidirectionalRNN
from NeuralNetwork.losses import * #(SparseCategoricalCrossEntropy, BinaryCrossEntropy,
                                  #CategoricalCrossEntropy)
from NeuralNetwork.optimizers import * #(StochasticGradientDescent, Adam,
                                      #Adagrad, RMSprop)
from NeuralNetwork.activations import * #Activation, Softmax, Sigmoid
from NeuralNetwork.model import NNetwork
from NeuralNetwork.metrics import *

##############################################################################

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


def start(epochs = 1000, learning_rate = 0.001, summary = True, model = 'a'):    

    if model == 'a':
        nn = NNetwork([
            Input(input_shape = train_x_ohe.shape[1:]),
            RNN(51, True),
            RNN(53),

            Linear(2),
            Softmax(),
        ])


    if model == 'b':
        nn = NNetwork([
                BidirectionalLSTM((50, 51)),
                GRU(53, True),
                BidirectionalGRU((21, 20)),
                LSTM(15, True),
                BidirectionalRNN((10, 15)),
                RNN(10, True),
                RNN(5),
                #Linear(1, 'sigmoid'),
                Linear(2),
                Softmax(),
                #Sigmoid(),
            ])
    
    
    if summary:
        nn.summary()
    
    nn.compile(
        loss = SparseCategoricalCrossEntropy(),
        metrics = SparseCategoricalAccuracy(),
        #loss = BinaryCrossEntropy(),
        #optimizer = StochasticGradientDescent(learning_rate,0.0, (-1, 1))
        #optimizer = GradientDescent(learning_rate, (-1, 1))
        #optimizer = RMSprop(learning_rate)
        #optimizer = Adagrad(learning_rate)
        optimizer = Adam(learning_rate)
    )
    
    losses = nn.fit(train_x_ohe, train_y, epochs = epochs, batch_size = 58)


    print((np.argmax(nn(train_x_ohe), axis = -1) == train_y[:, 0]).mean())
    print((np.argmax(nn(test_x_ohe), axis = -1) == test_y[:, 0]).mean())

    plt.plot(losses[0], label = 'Loss')
    plt.plot(losses[1], label = 'Accuracy')
    plt.legend()
    plt.show()
