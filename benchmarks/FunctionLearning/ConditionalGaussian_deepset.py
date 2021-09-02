import numpy as np
import torch
from models.SIREN import Siren,MLP
from models.DeepSets import DeepSets_Siren_Weight,DeepSets_Siren
from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer
from scipy.spatial import KDTree
import copy

"""
Fitting gaussian given mean and standard deviation
"""

np.random.seed(100)
def Gaussian(x,mean,sigma):
	return np.log(1./(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mean)/sigma)**2))

N_hyper = 1000
N_samples = 100

x_axis = np.linspace(-5,5,N_samples)
hyper_samples = np.random.uniform(size=(N_hyper,2))
hyper_samples[:,0] = hyper_samples[:,0]*10-5
hyper_samples[:,1] = hyper_samples[:,1]*3+1

x = []
hyper_x = []
y = []
for i in range(N_hyper):
	x.append(x_axis[:,None])
	hyper_x.append(np.repeat(hyper_samples[i:i+1],N_samples,axis=0))
	y.append(Gaussian(x_axis,hyper_samples[i,0],hyper_samples[i,1])[:,None])

n_sample_points = 5
n_hyper_points = 5

train_x = x[:int(N_hyper*0.8)]

sample_tree_array = []
for i in range(len(train_x)):
	sample_tree_array.append(KDTree(train_x[0]))
length = [0]
for i in range(len(train_x)):
	length.append(train_x[i].shape[0])
cum_length = np.cumsum(length)

train_hyper_x = hyper_x[:int(N_hyper*0.8)]
train_x = np.concatenate(train_x,axis=0)
train_hyper_x = np.concatenate(train_hyper_x,axis=0)

train_y = np.concatenate(y[:int(N_hyper*0.8)],axis=0)

valid_x = np.concatenate(x[int(N_hyper*0.8):],axis=0)
valid_hyper_x = np.concatenate(hyper_x[int(N_hyper*0.8):],axis=0)
valid_y = np.concatenate(y[int(N_hyper*0.8):],axis=0)


# np.unique() shuffles order of the array, I did this in order to match the indicies
hyper_tree = KDTree(train_hyper_x[np.sort(np.unique(train_hyper_x,axis=0,return_index=True)[1])])



def query_tree_array(samples, hyper, sample_tree_array, hyper_tree, n_sample_points=3, n_hyper_points=3, training_mode=False):
    if training_mode == True:
        hyper_query = hyper_tree.query(hyper,k=n_hyper_points+1)[1][:,1:]
    else:
        hyper_query = hyper_tree.query(hyper,k=n_hyper_points)[1]
    sample_query = np.repeat(samples.shape[0],n_hyper_points)
    output = np.empty((samples.shape[0],n_hyper_points,n_sample_points))
    for i in range(len(sample_tree_array)):
        if np.where(hyper_query==i)[0].size>0:
            if n_sample_points == 1:
                output[hyper_query==i] = sample_tree_array[i].query(samples[np.where(hyper_query==i)[0]],k=n_sample_points)[1][:,None]
            else:
                output[hyper_query==i] = sample_tree_array[i].query(samples[np.where(hyper_query==i)[0]],k=n_sample_points)[1]
    return output,hyper_query

train_index = query_tree_array(train_x,train_hyper_x, sample_tree_array, hyper_tree, n_sample_points,n_hyper_points,training_mode=True)
train_index = (cum_length[train_index[1]][...,None].repeat(5,axis=2)+train_index[0]).reshape(-1,n_hyper_points*n_sample_points).astype(int)


valid_index = query_tree_array(valid_x,valid_hyper_x, sample_tree_array, hyper_tree, n_sample_points,n_hyper_points,training_mode=True)
valid_index = (cum_length[valid_index[1]][...,None].repeat(5,axis=2)+valid_index[0]).reshape(-1,n_hyper_points*n_sample_points).astype(int)


train_x = torch.cat((torch.from_numpy(train_x),torch.from_numpy(train_hyper_x)),dim=1).cuda().float()
train_context = train_x[train_index]
train_y = torch.from_numpy(train_y).cuda().float()

valid_x = torch.cat((torch.from_numpy(valid_x),torch.from_numpy(valid_hyper_x)),dim=1).cuda().float()
valid_context = train_x[valid_index]
valid_y = torch.from_numpy(valid_y).cuda().float()


model = DeepSets_Siren_Weight(in_features=3, out_features=1, hidden_features=128, hidden_layers=3,set_features=50)
model.train().cuda().float()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)


batch_size = int(2e4+1)
valid_loss = 1e10
for epoch in range(30000):
#    print(epoch)
    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    running_loss = 0.0
    for index in range(indices.shape[0]//batch_size+1):
        locations = (train_context-train_x[:,None])[index*batch_size:(index+1)*batch_size]
        values = train_y[train_index][index*batch_size:(index+1)*batch_size]
        outputs = train_y[index*batch_size:(index+1)*batch_size]
        optimizer.zero_grad()

        pred_y = model(locations,values)
#        pred_y = model(inputs)
        loss = torch.mean(torch.sum(((pred_y-outputs)/(outputs))**2,dim=1))
        loss.backward()
    #    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()
        running_loss += loss.item()

    if epoch%1==0:
        with torch.no_grad():
            pred_y = model(valid_context-valid_x[:,None],train_y[valid_index])
            #pred_y = model(valid_x)
            valid_loss_temp = torch.mean(torch.sum(((pred_y-valid_y)/valid_y)**2,dim=1))
            if valid_loss_temp<=valid_loss:
                print("saving")
                print("Training loss is :"+str(running_loss))
                print("Validation loss is :"+str(valid_loss_temp))
                valid_loss = valid_loss_temp
                best_model = copy.deepcopy(model)

