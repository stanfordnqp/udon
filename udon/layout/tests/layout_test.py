import udon.layout as layout

import jax
import jax.numpy as np
import phidl
import unittest


def _make_phidl(rasterfn, raster_spec=(0, 0)):
  """Make a photonic "circuit" with a ring, waveguides and `RasterDevice`s."""
  # Describe a ring-coupled waveguide.
  ring = phidl.geometry.ring(radius=5, width=1)
  rect = phidl.geometry.rectangle(size=(5, 1))
  invd = layout.rasterdevice("rasta", raster_spec, rasterfn, ((0, 0), (5, 1)),
                             deraster_resolution=0.1)

  # Add ports on `rect`.
  rect.add_port(name='port1', midpoint=(0, 0.5), orientation=180)
  rect.add_port(name='port2', midpoint=(5, 0.5), orientation=0)

  # Add ports on `invd`.
  invd.add_port(name='port1', midpoint=(0, 0.5), orientation=180)
  invd.add_port(name='port2', midpoint=(5, 0.5), orientation=0)

  # Waveguide with invd in the middle.
  wg = phidl.Device("wg")
  rect_ref1 = wg.add_ref(rect)
  rect_ref2 = wg.add_ref(rect)
  invd_ref = wg.add_ref(invd)

  # Connect two `rect` and one `invd`.
  rect_ref1.connect('port2', invd_ref.ports['port1'])
  rect_ref2.connect('port1', invd_ref.ports['port2'])

  # Add macro ports to `wg` device.
  wg.add_port(rect_ref1.ports['port1'])
  wg.add_port(rect_ref2.ports['port2'])

  # Our device.
  rcwg = phidl.Device("wg_ring")

  wg_ref1 = rcwg.add_ref(wg)
  ring_ref = rcwg.add_ref(ring)
  wg_ref2 = rcwg.add_ref(wg)

  # Apply transformation to just one waveguide
  wg_ref2.mirror((0, 1))

  # Align the two devices with a spacing of 0.5.
  rcwg.distribute(direction='y', spacing=0.5)
  rcwg.align(alignment='x')

  rcwg.add_port(wg_ref1.ports['port1'])
  rcwg.add_port(wg_ref1.ports['port2'])

  # Perform a bunch of transformations.
  rcwg.rotate(angle=40)
  rcwg.move(destination=(-1, 0))
  rcwg.mirror()

  return rcwg


def _diag_sine(offset):
  """Simple raster function."""
  return lambda x, y: 0.5 * (np.sin(5 * (x + y)) + offset)


class TestLayout(unittest.TestCase):
  def test_deraster_quickplot(self):
    """Tests that we can plot derastered polygons."""
    phidl.quickplotter.quickplot(
        _make_phidl(_diag_sine(1), raster_spec=(0, 0)))

  def test_raster(self):
    """Tests that rastering works."""
    d = _make_phidl(_diag_sine(1), raster_spec=(0, 0))
    x, y = np.meshgrid(np.arange(-5, 15, 0.1), np.arange(-5, 15, 0.1))
    self.assertAlmostEqual(np.sum(layout.raster(d, (0, 0), x, y)), 5544.1113)

  def test_raster_no_derastered_polygons(self):
    """Tests rastering when `RasterDevice`s produce no derastered polygons."""
    d = _make_phidl(_diag_sine(-0.1), raster_spec=(0, 0))
    x, y = np.meshgrid(np.arange(-5, 15, 0.1), np.arange(-5, 15, 0.1))
    self.assertAlmostEqual(np.sum(layout.raster(d, (0, 0), x, y)), 5179.7607)

  def test_raster_no_raster_devices(self):
    """Tests rastering when no `RasterDevice` objects are selected."""
    d = _make_phidl(_diag_sine(1), raster_spec=(0, 1))
    x, y = np.meshgrid(np.arange(-5, 15, 0.1), np.arange(-5, 15, 0.1))
    self.assertAlmostEqual(np.sum(layout.raster(d, (0, 0), x, y)), 5046)

  def test_raster_no_polygons(self):
    """Tests rastering when no polygons are selected."""
    d = _make_phidl(_diag_sine(1), raster_spec=(0, 1))
    x, y = np.meshgrid(np.arange(-5, 15, 0.1), np.arange(-5, 15, 0.1))
    self.assertAlmostEqual(np.sum(layout.raster(d, (0, 1), x, y)), 498.11142)

  def test_gradient(self):
    def loss(offset):
      x, y = np.meshgrid(np.arange(-5, 15, 0.1), np.arange(-5, 15, 0.1))
      d = _make_phidl(_diag_sine(offset))
      return np.sum(layout.raster(d, (0, 0), x, y))

    loss_grad = jax.grad(loss)
    self.assertAlmostEqual(loss_grad(1.0), 500.5)

  def test_polygon_sdf(self):
      self.assertEqual(layout.polygon_sdf(
          np.array([(0, 0), (1, 1), (2, 0)]), np.array([-1]), np.array([0])), 1.)
      self.assertEqual(layout.polygon_sdf(
          np.array([(0, 0), (1, 1), (2, 0)]), np.array([1]), np.array([0.2])), -0.2)
      self.assertEqual(layout.polygon_sdf(
          np.array([(0, 0), (1, 1), (2, 0)]), np.array([1]), np.array([2])), 1.)

      if __name__ == '__main__':
          unittest.main()
