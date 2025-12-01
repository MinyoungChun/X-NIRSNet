import torch
from torch.utils.data import Dataset, DataLoader

class NIRS_Dataset(Dataset):
    def __init__(self, data_stream1, data_stream2, labels):
        # input shape: (N, 1, Ch, T)
        self.x1 = data_stream1
        self.x2 = data_stream2
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

def get_dummy_data(args):

    total_samples = args.num_subjects * args.trials_per_sub
    
    data_stream1 = torch.randn(total_samples, 1, args.channels, args.time_points)
    data_stream2 = torch.randn(total_samples, 1, args.channels, args.time_points)
    
    labels = torch.randint(0, 2, (total_samples,))
    
    subject_ids = torch.repeat_interleave(
        torch.arange(args.num_subjects), args.trials_per_sub
    )
    
    return data_stream1, data_stream2, labels, subject_ids

def get_loaders(args, test_sub_id):

    x1, x2, y, sub_ids = get_dummy_data(args)
    
    test_mask = (sub_ids == test_sub_id)
    train_mask = ~test_mask
    
    train_ds = NIRS_Dataset(x1[train_mask], x2[train_mask], y[train_mask])
    test_ds = NIRS_Dataset(x1[test_mask], x2[test_mask], y[test_mask])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, test_loader