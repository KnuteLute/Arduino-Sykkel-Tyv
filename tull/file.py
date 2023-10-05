from tull.utils import ASK, KNUT, JET, SIGURD, ANDRE

def load_dataset(path):
    folders = ['Ask', 'Knut', 'Jet', 'Sigurd', 'Vanlige_Ansikter_']
    if not (folders <= os.listdir(path)):
        raise Exception('expected different folder')

    dataset = {}
    for folder, id in zip(folders, [ASK, KNUT, KRISTOFFER, SIGURD, ANDRE]):
        for filepath in os.listdir(os.path.join(path, folder)):
            dataset.insert(id, filepath)

    return dataset
 
def save_weights(weights, filename, folder='models'):
    raise NotImplementedError()

def load_weights(filename, folder='models'):
    raise NotImplementedError()
    return [0, 0, 0]
