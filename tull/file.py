from tull.utils import ASK, KNUT, JET, SIGURD

def load_dataset(path):
    raise NotImplementedError()
    return {'a':['1', '2', '3'], 'b':['1', '2', '3'], 'c':['1', '2', '3'], }
 
def save_weights(weights, filename, folder='models'):
    raise NotImplementedError()

def load_weights(filename, folder='models'):
    raise NotImplementedError()
    return [0, 0, 0]
