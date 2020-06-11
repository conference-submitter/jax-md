# Copyright JAX MD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax.tree_util import register_pytree_node
from jax.lib import xla_bridge
import jax.numpy as np

import numpy as onp


i16 = np.int16
i32 = np.int32
i64 = np.int64

f32 = np.float32
f64 = np.float64


def static_cast(*xs):
  """Function to cast a value to the lowest dtype that can express it."""
  # NOTE(schsam): static_cast is so named because it cannot be jit.
  if xla_bridge.get_backend().platform == 'tpu':
    return (np.array(x, np.float32) for x in xs)
  else:
    return (np.array(x, dtype=onp.min_scalar_type(x)) for x in xs)


def register_pytree_namedtuple(cls):
  register_pytree_node(
      cls,
      lambda xs: (tuple(xs), None),
      lambda _, xs: cls(*xs))


def check_kwargs_time_dependence(kwargs):
  return

def check_kwargs_empty(kwargs):
  return


def merge_dicts(a, b):
  merged = dict(a)
  merged.update(b)
  return merged


def safe_mask(mask, fn, operand, placeholder=0):
  masked = np.where(mask, operand, 0)
  return np.where(mask, fn(masked), placeholder)
