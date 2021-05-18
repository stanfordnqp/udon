from udon.layout._device import bounded_raster

import jax
import numpy as onp
import phidl
from skimage.measure import find_contours


def deraster(rasterfn, bbox, resolution, threshold=0.5):
  """Converts `rasterfn` to `phidl`-compatible polygons.

  Args:
    rasterfn: `rasterfn(x, y)` computing raster values at points `(x, y)`.
    bbox: `((x0, y0), (x1, y1))`  bounding box at which polygons will be forced
        to terminate.
    resolution: The minimum resolution to use when sampling `rasterfn` to
        to find contours.
    threshold: Raster value to produce the contour at, defaults to `0.5`.

  Returns a list of polygons which can then be added to a `phidl.Device` via
  `add_polygon()`.
  """
  # Produce the raster values.
  xgrid = _gridpoints(bbox[0][0], bbox[1][0], resolution)
  ygrid = _gridpoints(bbox[0][1], bbox[1][1], resolution)
  x, y = onp.meshgrid(xgrid, ygrid, indexing='ij')
  rvals = bounded_raster(rasterfn, x, y, bbox)

  # Stop JAX from trying to trace the gradient through `deraster()` since we do
  # not intend to support differentiation here anyways.
  rvals = jax.lax.stop_gradient(rvals)

  # Produce de-embedded polygons by first finding contours and adding them to
  # an empty device via the `XOR` operation.
  d = phidl.Device()
  contours = _contour(xgrid, ygrid, onp.array(rvals), threshold)
  for c in contours:
    dnext = phidl.Device()
    dnext.add_polygon(c)
    d = phidl.geometry.xor_diff(d, dnext)

  # Intersect with rectangle that is the bounding box for crisp edges.
  bbox_device = phidl.Device()
  bbox_device.add_polygon([[bbox[0][0], bbox[0][1]],
                           [bbox[0][0], bbox[1][1]],
                           [bbox[1][0], bbox[1][1]],
                           [bbox[1][0], bbox[0][1]]])
  d = phidl.geometry.boolean(d, bbox_device, operation='and')

  # Throw away the device, and just return polygons.
  return d.get_polygons()


def _gridpoints(x0, x1, resolution):
  """Gridpoints encompassing `(x0, x1)` with resolution at least `resolution`.
  
  Also adds additional boundary points at both `x0 - dx` and `x1 - dx` where
  `dx` is the effective resolution.
  """
  length = x1 - x0
  n = _ceildiv(length, resolution)
  dx = length / n
  return onp.concatenate([onp.array([x0 - dx]),
                          onp.linspace(x0, x1, n, endpoint=True),
                          onp.array([x1 + dx])])


def _ceildiv(x, y):
  """Rounds `x/y` to next largest integer."""
  return -int(-x // y)


def _contour(x, y, z, threshold):
  """Similar to `pyplot.contour` but doesn't plot anything."""
  contours = []
  for c in find_contours(z, threshold):
    contours.append(
        onp.stack([_interp(c[:, 0], x), _interp(c[:, 1], y)], axis=-1))
  return contours


def _interp(x, xref):
  return onp.interp(x, onp.arange(len(xref)), xref)
