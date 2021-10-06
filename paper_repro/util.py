import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.15, type=float)
    parser.add_argument('--noise_multiplier', default=1.1, type=float)
    parser.add_argument('--l2_norm_clip', default=1.0, type=float)
    parser.add_argument('--batch_sizes', default=[256, 128, 64, 32, 16], nargs='+', type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--seed', default=42, type=int)

    return parser
