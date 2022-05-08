import numpy as np


# Weight Initializer Class
class WeightsInitializer(object):
    '''
        WeightsInitializer
            This Class helps in initializing weights.

        Args:
            shape:
                - type: tuple
                - about: Shape of the weight is defined here.

            **kwargs:
                    name:
                        - type: string
                        - about: Name of the weight is defined to identify
                                 weights uniquely.
                        - default: None

                    trainable:
                        - type: bool
                        - about: Define whether to make the weight trainable
                                 when used in a network.

                    initializer:
                        - type: string
                        - about: Appropriate Weight Initialization technique is defined.
                        - default: 'uniform'

                    dtype:
                        - type: string/numpy data type
                        - about: set the datatype of the weight.
                        - default: float32


        Note: Available weight initializers are:
                        - `zeros`
                        - `ones`
                        - `random_normal`
                        - `random_uniform`
                        - `uniform`
                        - `xavier_glorot_normal`
                        - `xavier_glorot_uniform`
                        - `he_normal`
                        - `he_uniform`
                        
                            
    '''
    def __init__(self, shape: tuple, **kwargs):
        self.shape = shape
        
        self.initializer = kwargs.get('initializer', 'uniform')
        self.trainable = kwargs.get('trainable', True)
        self.name = kwargs.get('name', None)
        self.__dtype = kwargs.get('dtype', np.float32)

        self.weight = self.__get_weights.astype(self.__dtype)

    @property
    def __get_weights(self):
        fan_in, fan_out = np.product(self.shape[:-1]), self.shape[-1]
        if self.initializer.lower() == 'zeros':
            return np.zeros(self.shape)

        elif self.initializer.lower() == 'ones':
            return np.ones(self.shape)
        
        elif self.initializer.lower() == 'random_normal':
            return np.random.normal(loc = 0.0, scale = 1.0, size = self.shape)
        
        elif self.initializer.lower() == 'random_uniform':
            return np.random.uniform(low = 0.0, high = 1.0, size = self.shape)
        
        elif self.initializer.lower() == 'uniform':
            return np.random.uniform(low = -np.sqrt(1/fan_in), high = np.sqrt(1/fan_in), size = self.shape)

        elif self.initializer.lower() == 'xavier_glorot_normal':
            return np.random.normal(loc = 0.0, scale = np.sqrt(2/(fan_in + fan_out)), size = self.shape)

        elif self.initializer.lower() == 'xavier_glorot_uniform':
            return np.random.uniform(low = -np.sqrt(6/(fan_in + fan_out)), high = np.sqrt(6/(fan_in + fan_out)), size = self.shape)
        
        elif self.initializer.lower() == 'he_normal':
            return np.random.normal(loc = 0.0, scale = np.sqrt(2/fan_in), size = self.shape)

        elif self.initializer.lower() == 'he_uniform':
            return np.random.uniform(low = -np.sqrt(6/fan_in), high = np.sqrt(6/fan_in), size = self.shape)

        else:
            raise Exception(f'`{self.initializer}` is not available. \nAvailable initializers are:\n',
                            '- `zeros` \n- `ones` \n- `random_normal` \n- `random_uniform` \n- `uniform`\n',
                            '- `xavier_glorot_normal` \n- `xavier_glorot_uniform` \n- `he_normal` \n- `he_uniform`')
            
    def __get__(self):
        return self.weight

    def __set__(self, weight):
        self.weight = weight

    def reshape(self, shape):
        self.__set__(self.weight.reshape(shape))

    def set_dtype(self, dtype):
        self.weight.astype(dtype)


        


        
