'''
Code for JAX implementations presented in: Enabling Fast
Differentially Private SGD via Just-in-Time Compilation and Vectorization
'''

import itertools
import time
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random, vmap
from jax.experimental import optimizers, stax
from jax.tree_util import tree_flatten, tree_multimap, tree_unflatten
from keras.utils.np_utils import to_categorical

from paper_repro.jax_resnet_impl import ResNet18

import data
import util as utils

def cifar_model(features, **_):
    out = hk.Conv2D(32, (3, 3), padding='SAME', stride=(1, 1))(features)
    out = jax.nn.relu(out)
    out = hk.Conv2D(32, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.AvgPool(2, strides=2, padding='VALID')(out)

    out = hk.Conv2D(64, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.Conv2D(64, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.AvgPool(2, strides=2, padding='VALID')(out)

    out = hk.Conv2D(128, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.Conv2D(128, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.AvgPool(2, strides=2, padding='VALID')(out)

    out = hk.Conv2D(256, (3, 3), padding='SAME', stride=(1, 1))(out)
    out = jax.nn.relu(out)
    out = hk.Conv2D(10, (3, 3), padding='SAME', stride=(1, 1))(out)

    return out.mean((1, 2))


def multiclass_loss(model, params, batch):
    inputs, targets = batch
    logits = model.apply(params, None, inputs)
    # convert the outputs to one hot shape according to the same shape as
    # logits for vectorized dot product
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    logits = stax.logsoftmax(logits)  # log normalize
    return -jnp.mean(jnp.sum(logits * one_hot, axis=-1))  # cross entropy loss

def clipped_grad(model, loss, params, l2_norm_clip, single_example_batch):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(partial(loss, model))(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads)


def private_grad(model, loss, params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
    """Return differentially private gradients for params, evaluated on batch."""
    clipped_grads = vmap(partial(clipped_grad, model, loss), (None, None, 0))(params, l2_norm_clip,
                                                                              batch)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    rngs = random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated_clipped_grads)
    ]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads
    ]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)

def partial_model(features):
    model = ResNet18(10)
    return model(features, is_training=True)

def main(args):
    data_fn = data.load_cifar10
    kwargs = {
        'batch_size': args.batch_size,
    }
    train_data_loader, _, num_train = data_fn(**kwargs)
    key = random.PRNGKey(args.seed)

    if args.model == "resnet18":
        model = hk.transform(partial_model)
    elif args.model == "cifar10":
        model = hk.transform(partial(cifar_model, args=args))
    else:
        raise RuntimeError(f"model argument can only be 'resnet18' or 'cifar10', got {args.model}")

    rng = jax.random.PRNGKey(42)
    dummy = np.transpose(iter(train_data_loader).next()[0].numpy(), (0, 2, 3, 1))
    init_params = model.init(key, dummy)
    opt_init, opt_update, get_params = optimizers.sgd(args.learning_rate)
    
    # differentially private update
    def private_update(rng, i, opt_state, batch):
        batch = (jnp.transpose(batch[0], (0,2,3,1)), batch[1])
        params = get_params(opt_state)
        rng = random.fold_in(rng, i)  # get new key for new random numbers
        return opt_update(
            i,
            private_grad(model, multiclass_loss, params, batch, rng, args.l2_norm_clip, args.noise_multiplier,
                    args.batch_size), opt_state)

    opt_state = opt_init(init_params)
    itercount = itertools.count()
    train_fn = jit(private_update)

    dummy = jnp.array(1.)

    timings = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        for i, batch in enumerate(train_data_loader):
            batch = (batch[0].numpy(), batch[1].numpy())
            opt_state = train_fn(
                key,
                next(itercount),
                opt_state,
                batch,
            )
            (dummy * dummy).block_until_ready()  # synchronize CUDA.
        duration = time.perf_counter() - start
        print("Time Taken: ", duration)
        timings.append(duration)

    print(timings)

if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument('--model')
    args = parser.parse_args()
    for batch_size in args.batch_sizes:
        print(batch_size)
        args.batch_size = batch_size
        main(args)
