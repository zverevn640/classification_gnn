import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # Optimisation arguments
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # Model arguments
    parser.add_argument('--d', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=32)
 

    # Experiment parameters
    parser.add_argument('--dataset', default='wisconsin')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--model', type=str, choices=["GCN", "GAT", "Sheaf"], default="Sheaf")

    return parser