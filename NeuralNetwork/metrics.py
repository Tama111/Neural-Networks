import numpy as np

class Accuracy(object):
    '''
        Accuracy (Base Class)
            This is a parent class for all metrics.
    '''
    def __call__(self, y_true, y_pred):
        raise NotImplementedError('')
    

class BinaryAccuracy(Accuracy):
    '''
        BinaryAccuracy (Base Class: Accuracy)

        - Mainly used for binary classification task.
    '''
    def __call__(self, y_true, y_pred):
        return np.sum(np.where(y_pred > 0.5, 1, 0) == y_true)/y_true.shape[0]

class CategoricalAccuracy(Accuracy):
    '''
        CategoricalAccuracy (Base Class: Accuracy)

        - Mainly used for multi-class classification task.
    '''
    def __call__(self, y_true, y_pred):
        if len(y_true.shape) == 2:
            return np.sum(np.argmax(y_pred, axis = -1) == np.argmax(y_true, axis = -1))/y_true.shape[0]
        elif len(y_true.shape) == 3:
            return np.sum(np.argmax(y_pred, axis = -1) == np.argmax(y_true, axis = -1))/np.prod(y_true.shape[:2])

class SparseCategoricalAccuracy(Accuracy):
    '''
        SparseCategoricalAccuracy (Base Class: Accuracy)

        - Can be used for both binary and multi-class classification task.
    '''
    def __call__(self, y_true, y_pred):
        if len(y_pred.shape) == 2:
            return np.sum(np.argmax(y_pred, axis = -1) == y_true[:, 0])/y_true.shape[0]
        elif len(y_pred.shape) == 3:
            return np.sum(np.argmax(y_pred, axis = -1) == y_true)/np.prod(y_true.shape[:2])

class R2Score(Accuracy):
    '''
        R2Score (Base Class: Accuracy)

        - Mainly used for regression task.
    '''
    def __call__(self, y_true, y_pred):
        nume = np.sum(np.square(y_true - y_pred))
        deno = np.sum(np.square(y_true - np.mean(y_true)))
        return 1 - nume/deno


