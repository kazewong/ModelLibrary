from torch.utils.data import Dataset
import h5py

class Data2VecDataset(Dataset):

        def __init__(self,
                path: str,
                transform=None):
            h5_file = h5py.File(path, 'r')
            self.transform = transform

            self.data: h5py.Dataset = h5_file['data'] # type: ignore

            assert isinstance(self.data, h5py.Dataset), "Data is not a dataset" 

            self.n_dim = len(self.data.shape[2:])

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            sample = self.data[index]
            if self.transform != None:
                for f in self.transform:
                    sample = f(sample)
            return sample

        def get_shape(self):
            return self[0].shape