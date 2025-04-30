from .zjumocap import ZJUMoCapDataset
from .people_snapshot import PeopleSnapshotDataset
from .zjumocapforview import ZJUMoCapDatasetforview

def load_dataset(cfg, split='train'):
    if split == 'view':
        dataset_dict = {
            'zjumocapforview': ZJUMoCapDatasetforview,
            'people_snapshot': PeopleSnapshotDataset,
        }
        return dataset_dict[cfg.name + 'forview'](cfg, split)
    else:
        dataset_dict = {
            'zjumocap': ZJUMoCapDataset,
            'people_snapshot': PeopleSnapshotDataset,
        }
        return dataset_dict[cfg.name](cfg, split)

