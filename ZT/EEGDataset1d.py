import torch
from torch.utils.data import Dataset
import numpy as np

class EEGDataset(Dataset):
	def __init__(self, X, y, transform=None):


		self.X = X
		self.y = y
		self.transform = transform

	def __len__(self):
		return len(self.y)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample = {'data': self.X[idx], 'labels': self.y[idx]}
		if self.transform:
			sample = self.transform(sample)
		
		return sample

