from torch.utils.data import Dataset
import h5py

class DiffusionDataset(Dataset):

        def __init__(self,
                path: str,
                transform=None,
                has_conditional=False):
            h5_file = h5py.File(path, 'r')
            self.has_conditional = has_conditional
            self.transform = transform

            self.data: h5py.Dataset = h5_file['data'] # type: ignore
            self.scale: float = h5_file.attrs['scale']
            self.std: float = h5_file.attrs['std']

            assert isinstance(self.data, h5py.Dataset), "Data is not a dataset" 

            self.conditional_data: h5py.Dataset = h5_file['conditional'] if has_conditional else None # type: ignore
            if has_conditional:
                assert isinstance(self.conditional_data, h5py.Dataset), "Conditional data is not a dataset"
                assert len(self.data) == len(self.conditional_data) , "Data length is not equal to conditional length"# type: ignore
            
            self.n_dim = len(self.data.shape[2:])

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            sample = self.data[index]
            if self.has_conditional:
                conditional = self.conditional_data[index]
                if self.transform != None:
                    for f in self.transform:
                        sample = f(sample)
                return sample, conditional
            else:
                if self.transform != None:
                    for f in self.transform:
                        sample = f(sample)
                return sample


        def get_shape(self):
            return self[0].shape

        def add_normalize(self):
            self.transform.append(lambda x: ((x-self.scale)/self.std))