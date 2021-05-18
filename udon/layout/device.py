from udon.layout._device import RasterDevice
from udon.layout.derasterization import deraster

import phidl


def rasterdevice(name, spec, rasterfn, bbox, deraster_resolution):
  """Returns a `RasterDevice` inside a `Device`.

  Args:
    name: Name given to the returned `phidl.Device`.
    spec: `(layer, datatype))` denoting the layer for the derastered polygons.
    rasterfn: `rasterfn(x, y)` returning raster values at `(x, y)`.
    bbox: `((x0, y0), (x1, y1))` bounding box for the `RasterDevice`.
    deraster_resolution: Minimum resolution to use when computing polygons.
  """
  # Add polygons to the `RasterDevice`.
  rd = RasterDevice(spec, rasterfn, bbox)
  polygons = deraster(rasterfn, bbox, deraster_resolution)
  for p in polygons:
    rd.add_polygon(p, spec)

  # Wrap the `RasterDevice` as a reference to a `phidl.Device`.
  d = phidl.Device(name)
  d.add_ref(rd)
  return d
