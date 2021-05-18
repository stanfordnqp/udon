import jax.numpy as np


def polygon_sdf(polygon, x, y):
  """Signed distance of `polygon` for `points`.

  Implementation of the algorithm described in 
  https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
  (under polygon), and https://www.shadertoy.com/view/wdBXRW, which computes
  "windings" to determine whether a point is inside a polygon.
 
  Args:
    polygon: is a `N x 2` array with of `(x, y)` pairs specifying the points
        that make up the polygon. The first and last points should not be
        equal as the connectivity of the polygon is assumed.
    x, y: arrays of points for which signed distances are to
        be computed, using the convention that negative/positive distances
        correspond to  interior/exterior of the polygon respectively.

  Returns signed distances as an array of shape equal to `x`.
  """
  points = np.stack([x, y], axis=-1)
  dist = np.linalg.norm(polygon[0] - points, axis=-1)
  sign = np.ones(points.shape[0:-1], points.dtype)

  for i in range(polygon.shape[0]):
    edge = polygon[i-1] - polygon[i]
    vect = points - polygon[i]

    # Update `sign`.
    wind = np.stack([points[..., 1] >= polygon[i, 1],
                     points[..., 1] < polygon[i-1, 1],
                     edge[0] * vect[..., 1] > edge[1] * vect[..., 0]], axis=-1)
    sign = np.where(
        np.all(wind, axis=-1) | np.all(np.logical_not(wind), axis=-1),
        -sign, sign)

    # Update `dist`.
    ratio = np.clip(np.dot(vect, edge) / np.dot(edge, edge), 0.0, 1.0)
    vect = vect - edge * np.expand_dims(ratio, axis=-1)
    dist = np.minimum(dist, np.linalg.norm(vect, axis=-1))

  return sign * dist
