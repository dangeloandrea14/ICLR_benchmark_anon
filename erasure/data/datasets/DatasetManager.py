from erasure.core.base import Configurable
from erasure.core.factory_base import *
from fractions import Fraction
import numpy as np
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from .Dataset import DatasetWrapper
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data

class DatasetManager(Configurable):

    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.partitions = {}

        self.datasource = global_ctx.factory.get_object( Local (self.params['DataSource']) )
        self.info(self.params['DataSource'])
        
        self.partitions['all'] = self.datasource.create_and_validate_data()

        self.ref_data_dict = {'all':'all'}
        
        self.parts_cfgs = self.params['partitions']
        #self.info(self.partitions['all'].data)
        self.batch_size=self.params['batch_size']
        self.name = self.datasource.get_name()

        #count number of classes in the dataset
        self.n_classes = self.partitions['all'].get_n_classes()

        self.add_partitions(self.parts_cfgs)

    def add_partitions(self, splits, postfix=""):
        for split in splits:
            self.add_partition(split,postfix)
            
        self.info(list(self.partitions.keys()))

    def add_partition(self, split, postfix = ""):
        split['parameters']['parts_names'] = [p + postfix for p in split['parameters']['parts_names']]

        splitted_data = get_instance_config(split)
        splitted_data.set_source(self.datasource)

        ##add dictionary to self of what partition refers to what data -> partition: ref_data
        self.ref_data_dict[splitted_data.parts_names[0]] = splitted_data.ref_data
        self.ref_data_dict[splitted_data.parts_names[1]] = splitted_data.ref_data

        self.partitions = splitted_data.split_data(self.partitions)

        self.info(list(self.partitions.keys()))
        for key, value in self.partitions.items():
            self.info(f"Partition {key} has {len(value)} samples")

    def get_loader_for_ids(self, list_ids):

        dataset = self.partitions['all']


        if isinstance(dataset[0][0],Data):
            main_loader = GeometricDataLoader(self.datasource.get_wrapper(Subset(dataset.data, list_ids)), batch_size=self.batch_size, drop_last=True)
        else:
            main_loader = DataLoader(
                self.datasource.get_wrapper(Subset(dataset.data, list_ids)),
                batch_size=self.batch_size, collate_fn = skip_nones_collate, shuffle=False, worker_init_fn = torch.initial_seed()
            )        

        return main_loader
           
           
    def get_loader_for(self, split_id, fold_fraction = None, drop_last=True):

        fold_fraction = None
        ref_data = self.ref_data_dict[split_id]

        dataset = self.datasource.get_wrapper(self.partitions['all'].data) if split_id == 'all' else self.datasource.get_wrapper(Subset(self.partitions['all'].data, self.partitions[split_id]))

        num_samples = len(dataset)

        if split_id == 'train':
            self.info(f"TRAINING WITH {num_samples} samples")

        if fold_fraction is not None:
            ##TODO
            pass

        else:
            if isinstance(dataset[0][0],Data):
                main_loader = GeometricDataLoader(dataset, batch_size=self.batch_size, drop_last=True)
            else:
                main_loader = DataLoader(dataset, batch_size=self.batch_size,  collate_fn = skip_nones_collate, shuffle=False, worker_init_fn = torch.initial_seed(),drop_last=drop_last)
            fold_loader = None

        return main_loader, fold_loader

    '''def revise_split(self, split_id, ids_list, additive=False):
        #TODO This method must be removed or updated ask to Andrea + Claudio
        #print(f"REVISING SPLIT {split_id} with {len(ids_list)} samples")
        #print(f"ids_list: {ids_list}")
        if not additive:
            #print(self.partitions[split_id])
            #print(ids_list)
            self.partitions[split_id] = [sample for sample in self.partitions[split_id] if sample not in ids_list]
        else:
            self.partitions[split_id] = list(set(self.partitions[split_id] + ids_list))'''

        
    def get_dataset_from_partition(self, split_id):
        """
        Returns the dataset corresponding to the given partition ID.

        Args:
            partition_id (str): The ID of the partition.

        Returns:
            Dataset: The dataset corresponding to the partition ID.
        """
        if split_id in self.partitions:
            return self.partitions['all'].data if split_id == 'all' else self.datasource.get_wrapper(Subset(self.partitions['all'].data, self.partitions[split_id])).data
        else:
            raise ValueError(f"Partition ID '{split_id}' not found in partitions.")
   

def skip_nones_collate(batch):
    batch = [item for item in batch if item is not None]
    return default_collate(batch)

