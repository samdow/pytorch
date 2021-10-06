import types
from typing import Mapping, Optional, Sequence, Union, Any

import jax
import jax.numpy as jnp

import haiku as hk

FloatStrOrBool = Union[str, float, bool]

class BlockV2(hk.Module):
  """ResNet V2 block without bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      gn_config: Mapping[str, FloatStrOrBool],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection

    gn_config = dict(gn_config)
    gn_config.setdefault("groups", min(32, channels))

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding=(0,0),
          name="shortcut_conv")
      self.proj_groupnorm = hk.GroupNorm(name="shortcut_groupnorm", **gn_config)

    conv_0 = hk.Conv2D(
        output_channels=channels,
        kernel_shape=3,
        stride=stride,
        with_bias=False,
        padding=(1,1),
        name="conv_0")

    gn_0 = hk.GroupNorm(name="groupnorm_0", **gn_config)

    conv_1 = hk.Conv2D(
        output_channels=channels,
        kernel_shape=3,
        stride=1,
        with_bias=False,
        padding=(1,1),
        name="conv_1")

    gn_1 = hk.GroupNorm(name="groupnorm_1", **gn_config)
    layers = ((conv_0, gn_0), (conv_1, gn_1))

    self.layers = layers

  def __call__(self, inputs, is_training, test_local_stats):
    x = shortcut = inputs

    if self.use_projection:
      shortcut = self.proj_conv(shortcut)
      shortcut = self.proj_groupnorm(shortcut)

    for i, (conv_i, gn_i) in enumerate(self.layers):
      x = conv_i(x)
      x = gn_i(x)
      if i < len(self.layers):
        x = jax.nn.relu(x)

    return jax.nn.relu(x + shortcut)

class BlockGroup(hk.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      gn_config: Mapping[str, FloatStrOrBool],
      use_projection: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    block_cls = BlockV2

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(channels=channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                    gn_config=gn_config,
                    name="block_%d" % (i)))

  def __call__(self, inputs, is_training, test_local_stats):
    out = inputs
    for block in self.blocks:
      out = block(out, is_training, test_local_stats)
    return out

def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")

class ResNet18(hk.Module):
    """ResNet18 model."""

    def __init__(
      self,
      num_classes: int,
      gn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      ):
        """Constructs a ResNet model.
        Args:

            num_classes: The number of classes to classify the inputs into.
            logits_config: A dictionary of keyword arguments for the logits layer.
            name: Name of the module.
            initial_conv_config: Keyword arguments passed to the constructor of the
            initial :class:`~haiku.Conv2D` module.
        """

        blocks_per_group=(2, 2, 2, 2)
        channels_per_group=(64, 128, 256, 512)
        use_projection=(False, True, True, True)

        gn_config = dict(gn_config or {})
        gn_config.setdefault("create_scale", True)
        gn_config.setdefault("create_offset", True)
        
        super().__init__(name=name)
        logits_config = dict(logits_config or {})
        logits_config.setdefault("w_init", jnp.zeros)
        logits_config.setdefault("name", "logits")

        # Number of blocks in each group for ResNet.
        check_length(4, blocks_per_group, "blocks_per_group")
        check_length(4, channels_per_group, "channels_per_group")

        initial_conv_config = dict(initial_conv_config or {})
        initial_conv_config.setdefault("output_channels", 64)
        initial_conv_config.setdefault("kernel_shape", 7)
        initial_conv_config.setdefault("stride", 2)
        initial_conv_config.setdefault("with_bias", False)
        initial_conv_config.setdefault("padding", (3,3))
        initial_conv_config.setdefault("name", "initial_conv")

        self.initial_conv = hk.Conv2D(**initial_conv_config)

        self.block_groups = []
        strides = (1, 2, 2, 2)
        for i in range(4):
          self.block_groups.append(
              BlockGroup(channels=channels_per_group[i],
                         num_blocks=blocks_per_group[i],
                         stride=strides[i],
                         gn_config=gn_config,
                         use_projection=use_projection[i],
                         name="block_group_%d" % (i)))
        self.final_groupnorm = hk.GroupNorm(name="final_groupnorm", groups=32, **gn_config)
        self.logits = hk.Linear(num_classes, **logits_config)

    def __call__(self, inputs, is_training, test_local_stats=False):
        out = inputs
        out = self.initial_conv(out)
        out = hk.max_pool(out,
                      window_shape=(3, 3, 1),
                      strides=(2, 2, 1),
                      padding="SAME")

        for block_group in self.block_groups:
            out = block_group(out, is_training, test_local_stats)


        out = self.final_groupnorm(out)
        out = jax.nn.relu(out)
        out = jnp.mean(out, axis=(0, 1))
        return self.logits(out)
