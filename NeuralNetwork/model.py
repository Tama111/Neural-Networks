import numpy as np
from warnings import warn


# Network
class NNetwork(object):
    '''
        NNetwork
            This is used to create a base network/ main model.


        Args:
            layers:
                - type: list
                - about: it consists of list of all the layers.
                - default: []

            **kwargs:
                    name:
                        - type: string
                        - about: Name of the network can be defined here.
                        - default: 'NNetwork'


        Note: more layers, can be added in the network using `.add()` method,
              where the argument is a layer.
            
    '''
    def __init__(self, layers: list = [], **kwargs):
        self.layers = layers

        self.name = kwargs.get('name', self.__class__.__name__)
        
        self.__initialize
        

    def add(self, layer):
        self.layers.append(layer)
        self.__initialize

    @property
    def __initialize(self):
        if len(self.layers):
            if self.layers[0].name.title().startswith('Input'):
                self.input_shape = self.layers[0].input_shape
                self.layers = self.layers[1:]
                self.trainable_weights = []

                out_shape = self.input_shape
                for layer in self.layers:

                    if hasattr(layer, 'build'):
                        try:
                            layer.build(out_shape)
                        except NotImplementedError as error:
                            layer.input_shape = out_shape

                    else:
                        layer.input_shape = out_shape

                    if hasattr(layer, 'output_shape'):
                        try:
                            out_shape = layer.output_shape
                            
                        except NotImplementedError as error:
                            print(f'Output shape of the layer `{layer.name}` should be defined.')

                    else:
                        layer.output_shape = out_shape

                    if hasattr(layer, 'trainable_weights'):
                        self.trainable_weights.append(layer.trainable_weights)

            else:
                warn('Input Layer is not defined, it may cause Some attributes to be not defined/initialized.')

            
    def summary(self, get_summary = False, show_summary = True, verbose = 1):
        total_params = 0
        trainable_params = 0

        info = {}
        for n, layer in enumerate(self.layers):

            info[f'{layer.name.lower()}_{n}'] = {}
            info[f'{layer.name.lower()}_{n}']['layer'] = layer.name
            info[f'{layer.name.lower()}_{n}']['input_shape'] = layer.input_shape
            info[f'{layer.name.lower()}_{n}']['output_shape'] = layer.output_shape

            if hasattr(layer, 'parameters'):
                total_params += layer.parameters    
                trainable_params += layer.trainable_parameters
                info[f'{layer.name.lower()}_{n}']['params'] = layer.parameters
                info[f'{layer.name.lower()}_{n}']['trainable_params'] = layer.trainable_parameters

            else:
                info[f'{layer.name.lower()}_{n}']['params'] = 0
                info[f'{layer.name.lower()}_{n}']['trainable_params'] = 0
                

        non_trainable_params = total_params - trainable_params

        if show_summary:

            if verbose == 0:
                print(f'{self.name.title()}')
                print('_' * 15)
                print(f'Total Params: {total_params}')
                print(f'Trainable Params: {trainable_params}')
                print(f'Non-Trainable Params: {non_trainable_params}')
                print('=' * 15, '\n')
                for n, (layer, layer_info) in enumerate(info.items()):
                    print(f'Layer: {n}')                    
                    print(f'Layer: {layer} ({layer_info["layer"]})')
                    print(f'Input Shape: {layer_info["input_shape"]}')
                    print(f'Output Shape: {layer_info["output_shape"]}')
                    print(f'Params: {layer_info["params"]}')
                    print(f'Trainable Params: {layer_info["trainable_params"]}')

                    print('_' * 15, '\n')

            if verbose == 1:
                print(f'{self.name.title()}')
                print('_'*15)
                print(f'Total Params: {total_params}')
                print(f'Trainable Params: {trainable_params}')
                print(f'Non-Trainable Params: {non_trainable_params}')
                print('='*100)
                print('  |Layer| \t |Input Shape| \t |Output Shape| \t|Params| \t |Trainable Params|')
                print('='*100, '\n')
                for n, (layer, layer_info) in enumerate(info.items()):
                    print(f'  |{layer_info["layer"]} ({layer})| \t |{layer_info["input_shape"]}| \t |{layer_info["output_shape"]}| \t |{layer_info["params"]}| \t |{layer_info["trainable_params"]}|')
                    print('_'*100)

                print('\n\n\n')
                    

        if get_summary:
            return info
        
        
    def compile(self, loss, optimizer, metrics = None):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        for layer in self.layers:
            if hasattr(layer, 'preset_optimizer'):
                try:
                    layer.preset_optimizer(optimizer)
                except NotImplementedError as error:
                    pass
                    
                
    def fit(self, X, y, epochs = 1, batch_size = 32, validation_data = None, verbose = 1):
        if not hasattr(self, 'loss'):
            raise Exception('Compile the network before fitting the data.')

        train_losses = []
        val_losses = []

        if self.metrics is not None:
            train_accuracies = []
            val_accuracies = []
        
        for e in range(epochs):
            train_loss = 0

            if self.metrics is not None:
                train_acc = 0
            if verbose > 1:
                print(f'Epoch {e} Starts...\n')
                print('Train Set:')
            for n, b in enumerate(range(0, X.shape[0], batch_size)):            
                pred = X[b:b+batch_size]
                y_true = y[b:b+batch_size]

                pred = self.__call__(pred)

                b_loss = self.loss(y_true, pred)/pred.shape[0]

                back_p = self.loss.backprop()
                for layer in self.layers[::-1]:
                    back_p = layer.backprop(back_p)

                train_loss += b_loss

                if self.metrics is not None:
                    b_acc = self.metrics(y_true, pred)
                    train_acc += b_acc
                    
                if verbose > 1:
                    if self.metrics is not None:
                        print(f'Batch: {b} \t Loss: {round(b_loss, 5)} \t Accuracy: {round(b_acc, 5)}')
                    else:
                        print(f'Batch: {b} \t Loss: {round(b_loss, 5)}')

            train_loss /= (n+1)
            train_losses.append(train_loss)

            if self.metrics is not None:
                train_acc /= (n+1)
                train_accuracies.append(train_acc)
            
            if validation_data is not None:
                if verbose > 1:
                    print('\nTest Set:')
                val_loss = 0

                if self.metrics is not None:
                    val_acc = 0
                    
                for n, b in enumerate(range(0, validation_data[0].shape[0], batch_size)):
                    val_pred = validation_data[0][b:b+batch_size]
                    val_true = validation_data[1][b:b+batch_size]
                    for layer in self.layers:
                        val_pred = layer(val_pred)

                    val_b_loss = self.loss(val_true, val_pred)/val_pred.shape[0]

                    val_loss += val_b_loss

                    if self.metrics is not None:
                        val_b_acc = self.metrics(val_true, val_pred)
                        val_acc += val_b_acc
                    if verbose > 1:
                        print(f'Batch: {b} \t Val Loss: {round(val_b_loss, 5)}')


                val_loss /= (n+1)
                val_losses.append(val_loss)

                if self.metrics is not None:
                    val_acc /= (n+1)
                    val_accuracies.append(val_acc)

            if verbose > 1:
                print('\n')
            
            if verbose > 0:
                if self.metrics is not None:
                    if validation_data is not None:                
                        print(f'Epoch: {e} \t loss: {round(train_loss, 5)} \t accuracy: {round(train_acc, 5)} \t val_loss: {round(val_loss, 5)} \t val_accuracy: {round(val_acc, 5)}')
                    else:
                        print(f'Epoch: {e} \t loss: {round(train_loss, 5)} \t accuracy: {round(train_acc, 5)}')
                else:
                    if validation_data is not None:                
                        print(f'Epoch: {e} \t loss: {round(train_loss, 5)} \t val_loss: {round(val_loss, 5)}')
                    else:
                        print(f'Epoch: {e} \t loss: {round(train_loss, 5)}')

            if verbose > 1: print('_'*100, '\n')


        if self.metrics is not None:
            if validation_data is not None:
                return train_losses, train_accuracies, val_losses, val_accuracies
            else:
                return train_losses, train_accuracies

        else:
            if validation_data is not None:
                return train_losses, val_losses
            else:
                return train_losses
                    
                
    def __call__(self, X):
        pred = X
        for layer in self.layers:
            pred = layer(pred)
            
        return pred



