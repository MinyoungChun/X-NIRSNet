import argparse
import torch
from dataset import get_loaders
from model import X_NIRSNet
from utils import set_seed, train_one_epoch, validate

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for test_sub_idx in range(args.num_subjects):
        print(f"--- Processing Fold: Subject {test_sub_idx} as Test ---")
        
        train_loader, test_loader = get_loaders(args, test_sub_id=test_sub_idx)

        model = X_NIRSNet(channels=args.channels, time_points=args.time_points).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            
        acc = validate(model, test_loader, device)
        print(f"Subject {test_sub_idx} Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-NIRSNet Training")
    
    parser.add_argument('--channels', type=int, default=13, help='Number of fNIRS channels')
    parser.add_argument('--fs', type=float, default=6.138, help='Sampling rate')
    parser.add_argument('--sec', type=int, default=60, help='Duration of data in seconds')
    parser.add_argument('--num_subjects', type=int, default=10, help='Total number of subjects')
    parser.add_argument('--trials_per_sub', type=int, default=20, help='Trials per subject')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    
    args.time_points = int(args.fs * args.sec) 
    
    main(args)