from udon.layout._device import RasterDevice
import collections

_ReferenceTransform = collections.namedtuple(
    "_ReferenceTransform",
    ["origin", "rotation", "x_reflection", "magnification"])


def _reftx(ref):
  return _ReferenceTransform(origin=ref.origin,
                             rotation=ref.rotation,
                             x_reflection=ref.x_reflection,
                             magnification=ref.magnification)


def extract_raster(device):
  """Extracts `RasterDevice` objects with their associated transformations.

  Returns a list of `(RasterDevice, [ReferenceTransform])` pairs where
  the values are an ordered list of transforms applied to the `RasterDevice`.
  """
  if isinstance(device, RasterDevice):
    return [(device, [])]

  z = []
  for ref in device.references:
    z.extend([(k, v + [_reftx(ref)]) for k, v in extract_raster(ref.parent)])
  return z


def filter_raster(device):
  """Returns a copy of `device` with all `RasterDevice` objects removed."""
  d = device.copy(device.name + "_frcopy", deep_copy=True)

  d.references[:] = [r for r in d.references
                     if not isinstance(r.parent, RasterDevice)]

  for ref in d.references:
    f = filter_raster(ref.parent)
    # Need to modify both `parent` (used by `phidl`) and `cell` (used by `gdspy`).
    ref.parent = f
    ref.ref_cell = f

  return d
