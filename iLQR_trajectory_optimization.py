import json

from ilqr.cost import PathQRCost
from ilqr.dynamics import BatchAutoDiffDynamics
from ilqr import iLQR
from matplotlib import pyplot as plt
import numpy as np
import theano
import theano.tensor as T


def load_trajectory_data(filepath):
  with open(filepath) as f:
    data = json.load(f)

    xs = []
    us = []
    for d in data['trajectoryData']:
      xs.append([d['xPos'], d['yPos'], d['heading'], d['speed']])
      us.append([d['accel'], d['omega']])
  return np.array(xs), np.array(us)


# Since heading data maybe detected flipped. We did a preprocess here to flip
# the heading by assuming heading should be continuous and more than 50% of
# heading are detected correct.
def preprocess_flip_heading(xs, us):
  N = len(xs)
  # Heading direction from input data
  data_dirs = np.c_[np.cos(xs[:, 2]), np.sin(xs[:, 2])]
  # Indices that break the continuous rule.
  data_discontinuous = np.sum(data_dirs[:-1] * data_dirs[1:], axis=1) < 0
  adj_flip_scalar = np.ones(N)
  adj_flip_scalar[1:][data_discontinuous] *= -1
  flip_scalar = np.cumprod(adj_flip_scalar)
  # Reverse flip_scalar based on assumption 2: more than 50% of heading are
  # detected correct.
  if np.sum(flip_scalar) < 0:
    flip_scalar *= -1
  need_flip = flip_scalar < 0
  # Flip the heading and speed, -PI < heading < PI
  xs[need_flip, 2] -= np.pi
  xs[xs[:,2] < -np.pi, 2] += np.pi
  xs[need_flip, 3] *= -1
  us[need_flip, 0] *= -1


class TrajectoryOptimizationDynamics(BatchAutoDiffDynamics):
  """Trajectory optimization auto-differentiated dynamics model."""

  def __init__(self, dt_list):
    """Init dynamics model.

    Args:
      dt_list: Time step list [s].

    Note:
      state: [pos_x, pos_y, sin(heading), cos(heading), velocity]
      action: [acceleration, angular_velocity]
      heading: 0 is pointing X+ axis and increasing clockwise.
    """
    def f(x, u, i):
      pos_x = x[..., 0]
      pos_y = x[..., 1]
      sin_heading = x[..., 2]
      cos_heading = x[..., 3]
      heading = T.arctan2(sin_heading, cos_heading)
      v = x[..., 4]

      a = u[..., 0]
      omega = u[..., 1]

      t_index = T.cast(i[..., 0], 'int32')
      dt = dt_list[t_index]

      # A = [[1, 0, 0, dt * cos(heading)],
      #      [0, 1, 0, dt * sin(heading)],
      #      [0, 0, 1, 0]
      #      [0, 0, 0, 1]]
      # B = [[0, 0],
      #      [0, 0],
      #      [0, dt],
      #      [dt, 0]]
      next_pos_x = pos_x + dt * cos_heading * v
      next_pos_y = pos_y + dt * sin_heading * v
      next_heading = heading + dt * omega
      next_v = v + dt * a

      # Return next state: Ax + Bu
      return T.stack([
        next_pos_x,
        next_pos_y,
        T.sin(next_heading),
        T.cos(next_heading),
        next_v
      ]).T

    super(TrajectoryOptimizationDynamics, self).__init__(f,
                                                         state_size=5,
                                                         action_size=2)

  @classmethod
  def augment_state(cls, state):
    """Augments angular state into a non-angular state by replacing heading
      with sin(heading) and cos(heading). In this case, it converts:
      [x, y, heading, v] -> [x, y, sin(heading), cos(heading), v]

    Args:
      state: State vector [reduced_state_size].

    Returns:
      Augmented state size [state_size].
    """
    if state.ndim == 1:
      x, y, heading, v = state
    else:
      x = state[..., 0].reshape(-1, 1)
      y = state[..., 1].reshape(-1, 1)
      heading = state[..., 2].reshape(-1, 1)
      v = state[..., 3].reshape(-1, 1)

    return np.hstack([x, y, np.sin(heading), np.cos(heading), v])

  @classmethod
  def reduce_state(cls, state):
    """Reduces a non-angular state into an angular state by replacing
      sin(heading) and cos(heading) with heading. In this case, it converts:
      [x, y, sin(heading), cos(heading), v] -> [x, y, heading, v]

    Args:
      state: Augmented state vector [state_size].

    Returns:
      Reduced state size [reduced_state_size].
    """
    if state.ndim == 1:
      x, y, sin_heading, cos_heading, v = state
    else:
      x = state[..., 0].reshape(-1, 1)
      y = state[..., 1].reshape(-1, 1)
      sin_heading = state[..., 2].reshape(-1, 1)
      cos_heading = state[..., 3].reshape(-1, 1)
      v = state[..., 4].reshape(-1, 1)

    heading = np.arctan2(sin_heading, cos_heading)
    return np.hstack([x, y, heading, v])


def plot_result(xs_in, us_in, xs_out, us_out):
  xs_calculated_by_us = [[xs_in[0,0], xs_in[0,1]]]
  for i in range(len(us_in)):
    pos_x = xs_calculated_by_us[i][0]
    pos_y = xs_calculated_by_us[i][1]
    heading = xs_in[i, 2]
    v = xs_in[i, 3]

    next_pos_x = pos_x + dt[i] * np.cos(heading) * v
    next_pos_y = pos_y + dt[i] * np.sin(heading) * v
    xs_calculated_by_us.append([next_pos_x, next_pos_y])
  xs_calculated_by_us = np.array(xs_calculated_by_us)

  v1 = []
  a1 = []
  h1 = []
  o1 = []
  for i in range(len(us_in)):
    v1.append(np.linalg.norm(xs_in[i+1, 0:2] - xs_in[i, 0:2]) / dt[i])
    a1.append((xs_in[i+1, 3] - xs_in[i, 3]) / dt[i])
    h1.append(np.arctan2(xs_in[i+1, 1] - xs_in[i, 1], xs_in[i+1, 0] - xs_in[i, 0]))
    o1.append((xs_in[i+1, 2] - xs_in[i, 2]) / dt[i])
  v1 = np.array(v1)
  a1 = np.array(a1)
  h1 = np.array(h1)
  o1 = np.array(o1)

  N = len(h1)
  data_dirs = np.c_[np.cos(h1), np.sin(h1)]
  data_discontinuous = np.sum(data_dirs[:-1] * data_dirs[1:], axis=1) < 0
  adj_flip_scalar = np.ones(N)
  adj_flip_scalar[1:][data_discontinuous] *= -1
  flip_scalar = np.cumprod(adj_flip_scalar)
  if np.sum(flip_scalar) < 0:
    flip_scalar *= -1
  need_flip = flip_scalar < 0
  h1[need_flip] -= np.pi
  h1[h1 < -np.pi] += np.pi
  v1[need_flip] *= -1

  plt.figure()
  plt.subplot(2,3,1)
  plt.title('origin pos, origin_integrated, optimization result')
  plt.plot(xs_in[:,0], xs_in[:,1], 'r.-', label='pos_in(m)')
  plt.plot(xs_out[:,0], xs_out[:,1], 'g.-', label='pos_out')
  plt.plot(xs_calculated_by_us[:,0], xs_calculated_by_us[:,1], 'y.-', label='pos by using start pos and integrate input control data')
  plt.legend()
  plt.subplot(2,3,2)
  plt.title('speed')
  plt.plot(np.arange(len(xs_in)), xs_in[:,3], 'r', label='speed_in(m/s)')
  plt.plot(np.arange(len(xs_in)), xs_out[:,3], 'g', label='speed_out')
  plt.plot(np.arange(len(v1)), v1, 'r:', label='derivative of pos_in')
  plt.legend()
  plt.subplot(2,3,3)
  plt.title('accel')
  plt.plot(np.arange(len(us_in)), us_in[:,0], 'r', label='accel_in(m/s2)')
  plt.plot(np.arange(len(us_in)), us_out[:,0], 'g', label='accel_out')
  plt.plot(np.arange(len(a1)), a1, 'r:', label='derivative of speed_in')
  plt.legend()
  plt.subplot(2,3,4)
  plt.title('heading')
  plt.plot(np.arange(len(xs_in)), xs_in[:,2], 'r', label='heading_in(rad)')
  plt.plot(np.arange(len(xs_in)), xs_out[:,2], 'g', label='heading_out')
  plt.plot(np.arange(len(h1)), h1, 'r:', label='derivative of pos_in')
  plt.legend()
  plt.ylim(-np.pi, np.pi)
  plt.subplot(2,3,5)
  plt.title('omega')
  plt.plot(np.arange(len(us_in)), us_in[:,1], 'r', label='omega_in(rad/s)')
  plt.plot(np.arange(len(us_in)), us_out[:,1], 'g', label='omega_out')
  plt.plot(np.arange(len(o1)), o1, 'r:', label='derivative of heading_in')
  plt.legend()
  plt.ylim(-np.pi, np.pi)
  plt.show()


if __name__ == '__main__':
  xs, us = load_trajectory_data('a_noisy_trajectory.json')
  preprocess_flip_heading(xs, us)
  dt = theano.shared(np.ones(len(xs)) * 0.1)  # Discrete time-steps in seconds

  # Compile the dynamics.
  dynamics = TrajectoryOptimizationDynamics(dt)
  xs_in = dynamics.augment_state(xs)
  us_in = us[:-1]
  # Cost function
  Q = np.diag([10, 10, 20/3, 20/3, 10/2])
  R = 0.1 * np.diag([1, 1])
  cost = PathQRCost(Q, R, xs_in, us_in)
  # iLQR optimization
  ilqr = iLQR(dynamics, cost, len(us_in))
  xs_out, us_out = ilqr.fit(xs_in[0], us_in)
  xs_out = dynamics.reduce_state(xs_out)

  dt = dt.get_value()
  plot_result(xs, us[:-1], xs_out, us_out)
  print('Done')
