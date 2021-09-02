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
	dims = mean.shape[0]
	distance = x-mean
	norm = 1./np.sqrt(np.linalg.det(sigma)*np.sqrt(2*np.pi)**(dims))
	nominator = np.einsum('ai,ai->a',np.einsum('ai,aij->aj',distance,np.linalg.inv(sigma)),distance)
	return norm*np.exp(-1/2*(nominator))

N_samples = 100000
mean = np.random.uniform(size=(1,2))
cov = np.repeat(np.eye(2)[None,:],N_samples,axis=0)

x = np.random.uniform(-5,5,size=(N_samples,2))
y = Gaussian(x,mean,cov)[:,None]

train_x = x[:int(x.shape[0]*0.9)]
valid_x = x[int(x.shape[0]*0.9):]
train_y = y[:int(x.shape[0]*0.9)]
valid_y = y[int(x.shape[0]*0.9):]

x_scaler = QuantileTransformer(n_quantiles=100)
x_scaler.fit(train_x)
#train_x = x_scaler.transform(train_x)
#valid_x = x_scaler.transform(valid_x)

y_scaler = StandardScaler()
y_scaler.fit(train_y)
#train_y = y_scaler.transform(train_y)
#valid_y = y_scaler.transform(valid_y)

train_x = torch.from_numpy(train_x).cuda().float()
train_y = torch.from_numpy(train_y).cuda().float()
valid_x = torch.from_numpy(valid_x).cuda().float()
valid_y = torch.from_numpy(valid_y).cuda().float()

n_points = 16
tree = KDTree(train_x.cpu())
train_context_coord = torch.from_numpy(tree.query(train_x.cpu(),k=n_points)[1]).cuda()
train_context_value = train_y[train_context_coord]

train_context = torch.cat((train_x[train_context_coord],train_context_value),dim=2)

valid_context_coord = torch.from_numpy(tree.query(valid_x.cpu(),k=n_points)[1]).cuda()
valid_context_value = train_y[valid_context_coord]
valid_context = torch.cat((train_x[valid_context_coord],valid_context_value),dim=2)

model = DeepSets_Siren_Weight(in_features=2, out_features=1, hidden_features=128, hidden_layers=3,set_features=50)
model.train().cuda().float()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-2)

batch_size = int(1e12)
valid_loss = 1e10
for epoch in range(30000):
#    print(epoch)
    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    running_loss = 0.0
    for index in range(indices.shape[0]//batch_size+1):
        inputs = train_context[index*batch_size:(index+1)*batch_size]
        outputs = train_y[index*batch_size:(index+1)*batch_size]
        optimizer.zero_grad()

        pred_y = model(inputs[:,:,:2]-train_x[:,None],inputs[:,:,2:])
#        pred_y = model(inputs)
        loss = torch.mean(torch.sum(((pred_y-outputs)/(outputs))**2,dim=1))
        loss.backward()
    #    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()
        running_loss += loss.item()

    if epoch%10==0:
        with torch.no_grad():
            pred_y = model(valid_context[:,:,:2]-valid_x[:,None],valid_context[:,:,2:])
            #pred_y = model(valid_x)
            valid_loss_temp = torch.mean(torch.sum(((pred_y-valid_y)/valid_y)**2,dim=1))
            if valid_loss_temp<=valid_loss:
                print("saving")
                print("Training loss is :"+str(running_loss))
                print("Validation loss is :"+str(valid_loss_temp))
                valid_loss = valid_loss_temp
                best_model = copy.deepcopy(model)
 
