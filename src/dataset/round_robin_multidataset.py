from torch.utils.data.dataset import Dataset
import numpy as np

class RoundRobinMultiDataset(Dataset):

    def __init__(self, dataset_collection : list):
        self._dataset_list = dataset_collection[:]  # slicing creates shallow copy
        
        dataset_lengths = list(map(lambda x: len(x), self._dataset_list))
        self._total_dataset_length = sum(dataset_lengths, 0)

        dataset_count = len(dataset_collection)
        self._indexmap = []     # indexmap -> tuple (dataset_id, index_id)
        dataset_indices = np.zeros(len(self._dataset_list), dtype = int)
        next_dataset_id = 0
        for _ in range(self._total_dataset_length):
            assigned = False
            while not assigned:
                dsindex = dataset_indices[next_dataset_id]
                if(dsindex < dataset_lengths[next_dataset_id]):
                    self._indexmap.append((next_dataset_id, dsindex))
                    dataset_indices[next_dataset_id] += 1
                    assigned = True
                next_dataset_id += 1
                next_dataset_id %= dataset_count

    def __len__(self):
        return self._total_dataset_length

    def  __getitem__(self,idx):
        index = self._indexmap[idx]
        return self._dataset_list[index[0]][index[1]]


