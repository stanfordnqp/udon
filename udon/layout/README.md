# Purpose

We want to connect the worlds of
- __conventional design__: waveguides, rings, gratings, and other standardized photonic components manually parameterized using _polygons_ in GDSII format, and
- __inverse design__: arbitrary shapes and even continuously varying densities represented in _rasterized_ form by numerical arrays,

This means being able to work with photonic structures using both a _boundary_ representation 
(i.e. polygons in conventional design) as well as a _rasterized_ representation 
(i.e. arrays with values `[0, 1]` as in inverse design).

The `udon-layout` package accomplishes this by starting with the `phidl` package,
which allows for convenient conventional design at scale,
and extending it with a `RasterDevice` object which allows for rasterized (and differentiable!)
photonic structures to be incorporated in a conventional photonic circuit design workflow.

# Example usage

```python
import phidl
from udon import layout

def make_phidl(rasterfn):
  """Designs the photonic circuit using `phidl` and `udon.layout`."""
	d = phid.Device()
  invd = layout.rasterdevice("invd", (0, 0), rasterfn, ((0, 0), (5, 1)),
                      deraster_resolution=0.1)
	d.add_ref(invd)
	# Add other conventional devices and arrange them...
  return d

def diagonal_sine(offset):
  """Describes the raster function for the inverse designed component."""
  return lambda x, y: 0.5 * (np.sin(5 * (x + y)) + offset)

# 1. Compute and plot the GDS of the circuit (incl. inverse design component).
qp(make_phidl(diag_sine(1)))

# 2. Rasterize the circuit and plot.
x, y = np.meshgrid(np.arange(-15, 15, 0.1), np.arange(-5, 12.5, 0.1))
plt.imshow(layout.raster(d, (0, 0), x, y))
plt.gca().invert_yaxis()

# 3. Compute the gradient of a loss function w.r.t. to the design parameter `offset`.
def loss(offset):
	x, y = np.meshgrid(np.arange(-15, 15, 0.1), np.arange(-5, 12.5, 0.1))
	d = make_phidl(diag_sine(offset))
	return np.sum(raster(d, (0, 0), x, y))

loss_grad = jax.grad(loss)
print("Gradient at offset=1: ", loss_grad(1.0))
```

# API 

## `layout.RasterDevice`

Far beyond just describing polygons, `phidl` provides a very convenient set of APIs
for building up photonic circuits from individual components.
The fundamental abstraction that `phidl` uses to accomplish this is the `Device` object
which is used to represent everything from a single waveguide section to the layout of an entire photonic circuit.

In order to allow inverse designed components to be integrated into this workflow,
we introduce the `RasterDevice` object which allows for a photonic component
to be defined by a rastered function `rasterfn(point)` which yields value within `[0, 1]`
instead of by a set of polygons where values above/below the threshold value of `0.5`
correspond to points inside/outside of the photonic structure.

Critically, a `RasterDevice` can undergo many of the transformations available to `Device` objects
such as translation, rotation, reflection, and magnification --
as well as being integrated into the workflow of connecting to and being grouped with `Device` objects.
Integration is not seamless however, as more advanced use cases where boolean operations
and other operations such as outlining are not yet fully supported.
Specifically, while the polygonal respresentation of a `RasterDevice` fully supports
these operations, some of these changes will not be represented in the rastered representation.

`RasterDevice` objects should be created via the following API

```python
rasterdevice(name, spec, rasterfn, bbox, deraster_resolution)
```

which returns a `phidl.Device` with a `RasterDevice` object included in it as a device reference.
The structure is named `name` and placed included in the GDS spec `spec = (layer, datatype)`.
It is confined to the bounding box `bbox = ((x0, y0), (x1, y1))` and will be derastered
(that is, its polygon form will be computed) with a resolution of at least `deraster_resolution`.

Note that both the `bbox`, `rasterfn`, and even `deraster_resolution` parameters
are given in the original reference frame of the device.
That is, as the photonic structure is moved, rotated, magnified, and/or reflected
its original definition and therefore its shape will _not_ change.
It is for this reason that we wrap a `RasterDevice` as a reference in a `phidl.Device`,
to discourage tampering the original definition of the `RasterDevice`
and instead modifying it only through transformations on it as a reference device.

## `layout.raster()`

The `udon.layout` package also exposes `raster()` which is used to convert a `phidl.Device`
into a rasterized representation.

Critically, `raster()` is designed to be differentiable so as to be amenable for optimization.
Along these lines, instead of producing arrays of rastered values, `raster()` instead deals
with functions of the form `f(x, y) --> r` where `r` are the rastered values.

This "functional" design allows for more flexibility in the rasterization process
in that it does not impose constraints on the rasterization resolution,
or even the area which is to be rasterized.
This may be very useful, for example, in a distributed setting where each node
may only need to process a fraction of the circuit of interest,
alleviating having to deal with large numerical arrays with correspondingly
large memory overheads.

## `layout.polygon_sdf()`

Lastly, `polygon_sdf()` is also exposed in order to allow for the computation
of signed distance functions for polygons, which can be useful for 3D visualization.

# Future extensions
While we have described is a simple, low-level API to allow for the integration of inverse designed (or raster parameterized) devices to be integrated in a traditional photonics circuit workflow, we anticipate a number of higher-level abstractions will immediately be useful such as 

- a waveguide whose thickness can be optimized,
- a photonic crystal whose holes can be resized/moved,
- a ring with variable radii and center position, and
- a "black box" which can take on arbitrary topology (subject to some minimum feature size constraint).

We look forward to building up a library of such components and implementing a basic library
of methods for them to interact with each other
since we believe that this would be an important new tool to enable the photonic circuit designers of tomorrow.
