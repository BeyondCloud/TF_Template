import numpy as np
from PIL import Image
import os
class DataGenerator:
    def __init__(self, config):
		self.config = config
		self.Ndata = 0
		#direct load data
		if self.config.load_mode == "":
		    x = np.random.random((500,1))*10
		    self.feats = x
		    self.labels = pow(x,3)
		    self.Ndata = len(x)

		elif self.config.load_mode == "path":
			csv_path = os.path.join('.','Playground','train.csv')
			self.paths,self.labels = self.get_paths_and_labels(csv_path)
			self.Ndata = len(self.labels)


    def next_batch(self, batch_size=1):
    	idx = np.random.choice(self.Ndata, batch_size)
    	#load and preprocess your data here
        
    	if self.config.load_mode == "":
        	yield self.feats[idx], self.labels[idx]
        elif self.config.load_mode == "path":
        	feats = np.zeros(tuple([batch_size])+tuple(self.config.state_size))
        	for i,p in enumerate(self.paths[idx]):
	        	feats[i,:] = np.array(Image.open(p)).reshape(1,-1)
	        yield feats, self.labels[idx].reshape(-1,1)


    def get_paths_and_labels(self,csv_name):
		p = []
		l = []
		with open(csv_name) as fop:
		    lines = fop.readlines()
		    for line in lines:
		    	pl = line.split(',')
		        p.append(pl[0])
		        l.append(pl[1])
		return np.array(p),np.array(l)