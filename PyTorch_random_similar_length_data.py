from torch.utils.data import Sampler
import random
from operator import itemgetter
from torch.utils.data import Dataset
from functools import partial
from typing import List, Optional, Tuple, Union


def to_list(arr):
    return(arr.tolist())


class ListDataset(Dataset):
    def __init__(self, 
                *data : List[torch.tensor]):
        """
        Args:
            data (list): (mulitple) lists each containing torch.tensors
        """
        
        self.dataset = list(zip(*data))
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

def pad_length_right_end(tensor, n_pad, value=0):
    
    return torch.nn.functional.pad(tensor, (0, 0, 0, n_pad), 'constant', value)

def collate_fn(batch):
    longest = max([len(x[0]) for x in batch])
    padded = [list(map(partial(pad_length_right_end, n_pad=longest-len(sample[0])), sample)) for sample in batch]
    padded = tuple(map(torch.stack, zip(*padded)))
    return padded


class BatchSamplerRandomSimilarLength(Sampler):
    """
    Batch sampler that samples indices of a dataset with simlar lengths
    
    Attributes: 
        batch_size (int) : size of resulting batches 
        super_batch_size (int) : size of the superset of indices that will be shuffled together 
        shuffle (bool) : Whether to shuffle batches 
        indices (List[Tuple[int, int]]) : Contains the the (index, length) of each datapoint in the dataset
        
        
    """
    def __init__(self, 
                 dataset : ListDataset, 
                 batch_size : int = 32, 
                 indices : List = None, 
                 shuffle : bool = True, 
                 superset_factor : int = 4):
        
        """
        Args: 
            dataset (ListDataset) : a dataset containing a List[Tuple[*torch.tensor]]
            batch_size (int) : batch size 
            super_batch_size (int) : size of the superset of indices that will respectively be shuffled  
            shuffle (bool) : Whether to shuffle batches 
            indices (List[Tuple[int, int]]) : Contains the the (index, length) of each datapoint in the dataset
        """
        self.batch_size = batch_size
        self.super_batch_size = superset_factor * batch_size
        self.shuffle = shuffle
        # get the indicies and length
        self.indices = [(i, len(item[0])) for i, item in enumerate(dataset)]
        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()
            
      
    def to_list(self, arr):
        return(arr.tolist())
    
    def __iter__(self):
        """
        Returns indices of the dataset batched by similar length. 
    
        Sorts indices by data length, divides them into a set of super-batches. 
        Shuffles each of the super-batches and then pends them together. 
        Then divides the indices into batches. 
        Because of this the length of the datapoinrs in the will be similar but each batch will contain some randomness
        """
        
        if self.shuffle:
            random.shuffle(self.indices)
        
        # split indices into super set of pooled by length 
        superset = tuple(np.array_split(np.array(sorted(self.indices, key=itemgetter(1))), (len(self.indices) // self.super_batch_size) + 1 ))
        # shuffle each of the supersets with map
        _ = list(map(np.random.shuffle, superset))
        # make one list again 
        pooled_indices = np.concatenate([i[:, 0] for i in superset])
        # get batches
        batches = np.array_split(pooled_indices, (len(self.indices) // self.batch_size) + 1 )
        # shuffle batchs 
        batches = list(map(self.to_list, batches))
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.indices) // self.batch_size
    
if __name__ == "__main__":   
    
    # example dataset: 
    x = []
    y = []
    for i in range(100):
        length = torch.randint(low=0, high=10000, size=(1, ))[0]
        x.append(torch.randint(low=0, high=10, size=(length, 3)))
        y.append(torch.randint(low=0, high=10, size=(length, 1)))
        
    dataset = ListDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, 
                                        batch_sampler = BatchSamplerRandomSimilarLength(dataset, 10, superset_factor=2, shuffle=True))
