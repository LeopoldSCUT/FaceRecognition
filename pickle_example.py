import pickle
import numpy as np

def load_feature(inputfile = 'example.pkl'):
    with open(inputfile, 'rb') as f:
        feature = pickle.load(f)
    return feature

def save_feature(feature, outputfile = 'example.pkl'):
    with open(outputfile, 'wb') as f:
        pickle.dump(feature, f)
    return

if __name__=="__main__":
    #feat = np.zeros([34512, 100])
    #save_feature(feat)
    feat_load = load_feature('./version2.pkl')
    print(feat_load.shape)
    print(feat_load)