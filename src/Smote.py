import numpy as np
import random
import matplotlib.pyplot as plt

class SMOTE(object):
    def __init__(self,sample,k=2,gen_num=3):
        self.sample = sample      
        self.sample_num,self.feature_len = self.sample.shape
        self.k = min(k,self.sample_num-1)                
        self.gen_num = gen_num    
        self.syn_data = np.zeros((self.gen_num,self.feature_len))  
        self.k_neighbor = np.zeros((self.sample_num,self.k),dtype=int)  

    def get_neighbor_point(self):
        for index,single_signal in enumerate(self.sample):
            Euclidean_distance = np.array([np.sum(np.square(single_signal-i)) for i in self.sample])
            Euclidean_distance_index = Euclidean_distance.argsort()
            self.k_neighbor[index] = Euclidean_distance_index[1:self.k+1]

    def get_syn_data(self):
        self.get_neighbor_point()
        for i in range(self.gen_num):
            key = random.randint(0,self.sample_num-1)
            K_neighbor_point = self.k_neighbor[key][random.randint(0,self.k-1)]
            gap = self.sample[K_neighbor_point] - self.sample[key]
            self.syn_data[i] = self.sample[key] + random.uniform(0,1)*gap
        return self.syn_data

