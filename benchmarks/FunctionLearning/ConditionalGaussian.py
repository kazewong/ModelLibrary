import numpy as np
import torch
from models.SIREN import Siren,MLP
from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer
from scipy.spatial import KDTree

"""
Fitting gaussian given mean and standard deviation
"""

np.random.seed(100)
def Gaussian(x,mean,sigma):
	return np.log(1./(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mean)/sigma)**2))

N_hyper = 100
N_samples = 10000

x_axis = np.linspace(-20,20,N_samples)
hyper_samples = np.random.uniform(size=(N_hyper,2))
hyper_samples[:,0] = hyper_samples[:,0]*10-5
hyper_samples[:,1] = hyper_samples[:,1]*3+1

x = np.empty((0,3))
y = np.empty((0,1))
for i in range(N_hyper):
	x = np.append(x,np.append(x_axis[:,None],np.repeat(hyper_samples[i:i+1],N_samples,axis=0),axis=1),axis=0)
	y = np.append(y,Gaussian(x_axis,hyper_samples[i,0],hyper_samples[i,1])[:,None],axis=0)


#random_index = np.arange(N_hyper*N_samples)
#np.random.shuffle(random_index)

train_x = x[:int(x.shape[0]*0.9)]
valid_x = x[int(x.shape[0]*0.9):]
train_y = y[:int(x.shape[0]*0.9)]
valid_y = y[int(x.shape[0]*0.9):]

x_scaler = QuantileTransformer(n_quantiles=100)
x_scaler.fit(train_x)
train_x = x_scaler.transform(train_x)
valid_x = x_scaler.transform(valid_x)

y_scaler = StandardScaler()
y_scaler.fit(train_y)
#train_y = y_scaler.transform(train_y)
#valid_y = y_scaler.transform(valid_y)


train_x = torch.from_numpy(train_x).cuda().float()
train_y = torch.from_numpy(train_y).cuda().float()
valid_x = torch.from_numpy(valid_x).cuda().float()
valid_y = torch.from_numpy(valid_y).cuda().float()

model = Siren(in_features=3, out_features=1, hidden_features=128,
              hidden_layers=3, outermost_linear=True)
model.train().cuda().float()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

batch_size = int(1e8)
valid_loss = 1e10
for epoch in range(30000):
#    print(epoch)
    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    running_loss = 0.0
    for index in range(indices.shape[0]//batch_size+1):
        inputs = train_x[index*batch_size:(index+1)*batch_size]
        outputs = train_y[index*batch_size:(index+1)*batch_size]
        optimizer.zero_grad()

        pred_y,coords = model(inputs)
#        pred_y = model(inputs)
        loss = torch.mean(torch.sum((pred_y[:,:2]-outputs)**2,dim=1))
        loss.backward()
    #    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()
        running_loss += loss.item()

    if epoch%10==0:
        with torch.no_grad():
            pred_y,coords = model(valid_x)
            #pred_y = model(valid_x)
            valid_loss_temp = torch.mean(torch.sum((pred_y[:,:2]-valid_y)**2,dim=1))
            if valid_loss_temp<=valid_loss:
                print("saving")
                print("Training loss is :"+str(running_loss))
                print("Validation loss is :"+str(valid_loss_temp))
                valid_loss = valid_loss_temp

 
