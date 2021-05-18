import phidl
import jax.numpy as np


class RasterDevice(phidl.Device):
  """Device described by a raster function.

  Strongly recommended to use `rasterdevice()` instead of directly instantiating
  here, since `rasterdevice()` will insert a `RasterDevice` as a reference
  inside of a `phidl.Device` object.

  This helps prevent arbitrary operations (some of which may not be supported)
  one a `RasterDevice`.
  """

  def __init__(self, spec, rasterfn, bbox, *args, **kwargs):
    super(RasterDevice, self).__init__(*args, **kwargs)
    self._bbox = bbox
    self._rasterfn = rasterfn
    self._spec = spec

  @property
  def bbox(self):
    return self._bbox

  # Needed so that even if we don't have any polygons, we can still detect the
  # spec, which is needed when rasterizing.
  @property
  def spec(self):
    return self._spec

  def raster(self, x, y):
    return bounded_raster(self._rasterfn, x, y, self.bbox)


def bounded_raster(rasterfn, x, y, bbox):
  """Returns `rasterfn(x, y)` clipped to `[0, 1]` if in `bbox`, `0` otherwise."""
  in_bbox = np.logical_and(
      np.logical_and(x >= bbox[0][0], x <= bbox[1][0]),
      np.logical_and(y >= bbox[0][1], y <= bbox[1][1]))
  return np.where(in_bbox, np.clip(rasterfn(x, y), 0., 1.), 0.0)
