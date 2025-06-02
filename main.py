import numpy as np
import torch
import torch.nn.functional as F
import random

from tqdm import tqdm
from utility.utils import local_pca_alignment
from models.contenders import ContenderModel
from models.sheaf_diffusion import SheafDiffusion
from parser import get_parser
from graph_datasets import get_dataset

def get_fixed_splits(data, dataset_name, seed):
    with np.load(f'splits/{dataset_name}_split_0.6_0.2_{seed}.npz') as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        data.train_mask[data.non_valid_samples] = False
        data.test_mask[data.non_valid_samples] = False
        data.val_mask[data.non_valid_samples] = False
        print("Non zero masks", torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
        print("Nodes", data.x.size(0))
        print("Non valid", len(data.non_valid_samples))
    else:
        assert torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask) == data.x.size(0)

    return data

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()

    optimizer.step()
    del out


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses
    
def run(args, dataset, model_cls, fold, conn_lap):
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold)
    
    data = data.to(args['device'])


    model = model_cls(data.edge_index, args, conn_lap)
    # model = model_cls(data.edge_index, args)
    model = model.to(args['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    epoch = 0
    best_val_acc = test_acc = 0
    best_epoch = 0

    for epoch in range(args['epochs']):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        new_best_trigger = val_acc > best_val_acc 
        if new_best_trigger:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_epoch = epoch

            
    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.3f}")
    print(f"Best val acc: {best_val_acc:.3f}")

    print({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})

    return test_acc, best_val_acc

if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()
    dataset = get_dataset(args.dataset)
    
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    print(args)
    conf = vars(args)
    
    if args.model == "Sheaf":
        model_cls = SheafDiffusion
    else:
        model_cls = ContenderModel

    data = dataset[0]
    # data = data.to(conf['device'])
    conn_lap = None
    conn_lap = local_pca_alignment(np.array(data.x), data.edge_index, stalk_dim=conf["d"])
    conn_lap = conn_lap.to(conf["device"])
    for fold in tqdm(range(args.folds)):
        test_acc, best_val_acc = run(conf, dataset, model_cls, fold, conn_lap)
        results.append([test_acc, best_val_acc])

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    print(f'{args.model} on {args.dataset}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')