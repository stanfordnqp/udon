from udon.layout.polygon_signed_distance import polygon_sdf
from udon.layout.device_parse import filter_raster, extract_raster

import jax.numpy as np


def raster(device, spec, x, y):
  """Returns raster values at points `(x, y)` of `device` components at `spec`.

  Args:
    device: `phidl.Device` in which `RasterDevice` objects may be incorporated.
    spec: `(layer, datatype)` of components to be included in the rasterization.
    x: x-position of points at which to return raster values.
    y: y-position of points at which to return raster values.

  Returns:
    `jax.DeviceArray` of raster values of the same shape as `x`.
  """
  x, y = np.broadcast_arrays(x, y)
  poly_rvals = _polygon_raster(filter_raster(device), spec, x, y)
  rdev_rvals = _device_raster(device, spec, x, y)
  return np.max(np.stack([poly_rvals, rdev_rvals]), axis=0)


def _polygon_raster(device, spec, x, y):
  """Raster values for polygons."""
  polygons = device.get_polygons(by_spec=spec)
  if polygons:
    rvals = np.stack([polygon_sdf(p, x, y) <= 0.0 for p in polygons])
    return np.any(rvals, axis=0)
  else:
    return np.zeros_like(x)


def _device_raster(device, spec, x, y):
  """Raster values for `RasterDevice` objects."""
  rvals = [_transformed_raster(d, txs, x, y)
           for d, txs in extract_raster(device) if spec == d.spec]
  if rvals:
    return np.max(np.stack(rvals), axis=0)
  else:
    return np.zeros_like(x)


def _transformed_raster(rdevice, transforms, x, y):
  """Apply `transforms` to `(x, y)` points and rasters the `RasterDevice`."""
  for t in transforms:
    x, y = _spatial_transform(t, x, y)
  return rdevice.raster(x, y)


def _spatial_transform(transform, x, y):
  """Transform points `(x, y)` according to the `transform`.

  Can be thought of as the dual to the transformations applied to polygons as
  performed in `CellReference._transform_polygons()` in the `gdspy` package,
  see https://github.com/heitzmann/gdspy/blob/b92a8fb9574342d895cf600f0c5460178578534c/gdspy/library.py#L1253.
  That is, instead of transforming the polygons, we transform the spatial
  coordinates in which rasterization occurs.

  Args:
    transform: `_ReferenceTransform` object.
    x, y: Arrays denoting the points to transform.
  
  Returns transformed `(x, y)` points.
  """
  x = x - transform.origin[0]
  y = y - transform.origin[1]

  theta = -transform.rotation * np.pi/180
  x, y = (x * np.cos(theta) - y * np.sin(theta),
          y * np.cos(theta) + x * np.sin(theta))

  if transform.magnification is not None:
    x = x / transform.magnification
    y = y / transform.magnification

  if transform.x_reflection:
     y = -y

  return x, y
