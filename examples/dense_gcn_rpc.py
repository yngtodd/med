import os

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc import rpc_sync
from torch.distributed.optim import DistributedOptimizer

from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import DenseGCNConv


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _parameter_rrefs(module):
    r"""
    Create one RRef for each parameter in the given local module, and return a
    list of RRefs.
    """
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs


class GATin(torch.nn.Module):
    """ In block of the GATNet """

    def __init__(self, in_channels, out_channels):
        super(GATin, self).__init__()
        self.conv = DenseGCNConv(in_channels, out_channels) 

    def forward(self, x, data_flow):
        block = data_flow[0]
        x = x[block.n_id]
        x = F.elu(self.conv(
            (x, x[block.res_n_id]), block.edge_index, size=block.size)
        )
        x = F.dropout(x, p=0.6, training=self.training)
        block = data_flow[1]
        return x, block


class GATout(torch.nn.Module):
    """ Out block of the GATNet """

    def __init__(self, in_channels, out_channels):
        super(GATout, self).__init__()
        self.conv = DenseGCNConv(in_channels, out_channels)

    def forward(self, x, block):
        x = self.conv(
            (x, x[block.res_n_id]), block.edge_index, size=block.size
        )
        return F.log_softmax(x, dim=1)


class GRPCNet(torch.nn.Module):
    """ Model parallel GCN  """

    def __init__(self, ps, in_channels, out_channels):
        super(GRPCNet, self).__init__()
        self.gat_in = GATin(in_channels, out_channels)
        self.gat_out_rref = rpc.remote(ps, GATout, args=(out_channels, out_channels))

    def forward(self, x, data_flow):
        x, block = self.gat_in(x, data_flow)
        x = _remote_method(GATout.forward, self.gat_out_rref, x, block)
        return x

    def parameter_rrefs(self):
        """ Hold the parameter references for the model """
        remote_params = []
        # create RRefs for local parameters
        remote_params.extend(_parameter_rrefs(self.gat_in))
        print('I have the gat_in params')
        # get RRefs of the GATout module 
        remote_params.extend(_remote_method(_parameter_rrefs, self.gat_out_rref))
        print('I have the gat_out params')
        return remote_params


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()


def test(mask, model, loader, device):
    model.eval()
    correct = 0
    for data_flow in loader(mask):
        pred = model(data.x.to(device), data_flow.to(device)).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
    return correct / mask.sum().item()


def run_trainer():

    dataset = Reddit('~/data')
    data = dataset[0]
    loader = NeighborSampler(
        data, size=[25, 10], num_hops=2, batch_size=1000, 
        shuffle=True, add_self_loops=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRPCNet("ps", dataset.num_features, dataset.num_classes).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # setup distributed optimizer
    optimizer = DistributedOptimizer(
        torch.optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    for epoch in range(1, 31):
        loss = train(model, loader, optimizer, device)
        test_acc = test(data.test_mask, model, loader, device)
        print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(
              epoch, loss, test_acc))


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 1:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)
