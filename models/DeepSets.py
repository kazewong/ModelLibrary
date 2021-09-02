import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.SIREN import Siren
from models.MLP import MLP

class DeepSets_Siren(torch.nn.Module):

	def __init__(self, in_features, hidden_features, hidden_layers, out_features, set_features=50):
		super().__init__()
		self.feature_extractor = Siren(in_features+out_features, hidden_features, hidden_layers, set_features, outermost_linear=True)
		self.query_extractor = Siren(in_features, hidden_features, hidden_layers, set_features, outermost_linear=True)
		self.regressor = Siren(set_features, hidden_features, hidden_layers, out_features, outermost_linear=True)

	def forward(self, context):#, query):
		context,context_coord = self.feature_extractor(context)
		pooled_context = context.sum(dim=1)
		context_query = pooled_context
		output,output_coord = self.regressor(context_query)
		return output,output_coord

class DeepSets_Siren_Weight(torch.nn.Module):

	def __init__(self, in_features, hidden_features, hidden_layers, out_features, set_features=50):
		super().__init__()
		self.feature_extractor = Siren(in_features+out_features, hidden_features, hidden_layers, set_features, outermost_linear=True)
		self.regressor = Siren(set_features, hidden_features, hidden_layers, out_features, outermost_linear=True)

	def get_weight(self,locations, values):
		context = torch.cat((locations, values), axis=-1)
		context,context_coord = self.feature_extractor(context)
#		context,context_coord = self.regressor(context)

		pooled_context = context.sum(dim=2)
		pooled_context = F.softmax(pooled_context,dim=1)
		return pooled_context
	

	def forward(self, locations, values):#, query):
		weights = self.get_weight(locations,values)
		output = (weights[:,:,None]*values).sum(dim=1)
		return output



class DeepSets(torch.nn.Module):

	def __init__(self, in_features, hidden_features, hidden_layers, out_features, set_features=50):
		super().__init__()
		self.feature_extractor = MLP(in_features+out_features, set_features, hidden_features, hidden_layers)
		self.regressor = MLP(set_features, out_features, hidden_features, hidden_layers)

	def forward(self, context):#, query):
		context = self.feature_extractor(context)
		pooled_context = context.sum(dim=1)
		output = self.regressor(pooled_context)
		return output
