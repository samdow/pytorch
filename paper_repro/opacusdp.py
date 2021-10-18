from opacus import PrivacyEngine
import torch
from torch import nn, optim
import torchvision.models as models
from opacus.utils.module_modification import convert_batchnorm_modules

import time
import numpy as np

import paper_repro.util as utils
import paper_repro.data as data

class CIFAR10Model(nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.layer_list = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32, 32, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(32, 64, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64, 64, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(64, 128, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.AvgPool2d(2, stride=2),
            nn.Sequential(nn.Conv2d(128, 256, (3, 3), padding=1, stride=(1, 1)), nn.ReLU()),
            nn.Conv2d(256, 10, (3, 3), padding=1, stride=(1, 1)),
        ])
    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return torch.mean(x, dim=(2, 3))

def train_opacus(args, model, train_loader, optimizer, device):
    for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            outputs = model.forward(x)
            loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize() # synchronize cuda

def get_data(args):
    data_fn = data.load_cifar10
    kwargs = {
        'batch_size': args.batch_size,
        'format': 'NCHW',
    }
    train_data_loader, _, sample_size = data_fn(**kwargs)
    return train_data_loader, sample_size

def main(args):
    train_data_loader, sample_size = get_data(args)

    if args.model == "cifar10":
        model = CIFAR10Model(batch_size=args.batch_size).cuda()
    elif args.model == "resnet18":
        model = convert_batchnorm_modules(models.resnet18(num_classes=10)).cuda()
    else:
        raise RuntimeError(f"only supported models are 'cifar10' or 'resnet18' got {args.model}")

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=args.batch_size,
        sample_size=sample_size,
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=args.sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
    )
    privacy_engine.attach(optimizer)

    timings = []
    before = torch.cuda.memory_allocated()
    print(args.batch_size)
    for epoch in range(args.epochs):
        start = time.perf_counter()
        train_opacus(args, model, train_data_loader, optimizer, torch.device("cuda"))
        duration = time.perf_counter() - start
        print("Time Taken for Epoch: ", duration)
        timings.append(duration)

    print(timings)


if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument("--model")
    args = parser.parse_args()

    for batch_size in args.batch_sizes:
        args.batch_size = batch_size
        main(args)
