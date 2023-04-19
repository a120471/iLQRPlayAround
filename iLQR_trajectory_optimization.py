from functools import partial
import json

import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from trajax import optimizers


def load_trajectory_data(filepath):
  with open(filepath) as f:
    data = json.load(f)

    xs = []
    us = []
    for d in data['trajectoryData']:
      xs.append([d['xPos'], d['yPos'], d['heading'], d['speed']])
      us.append([d['accel'], d['omega']])
  return np.array(xs), np.array(us)


# Since heading data maybe detected flipped, we did a preprocess here to flip
# the heading by assuming heading should be continuous and more than 50% of
# heading are detected correct.
def preprocess_flip_heading(xs, us):
  N = len(xs)
  # Heading direction from input data
  data_dirs = np.c_[np.cos(xs[:, 2]), np.sin(xs[:, 2])]
  # Indices that break the continuous assumption.
  data_discontinuous = np.sum(data_dirs[:-1] * data_dirs[1:], axis=1) < 0
  adj_flip_scalar = np.ones(N)
  adj_flip_scalar[1:][data_discontinuous] *= -1
  flip_scalar = np.cumprod(adj_flip_scalar)
  # Reverse flip_scalar based on the 2nd assumption: more than 50% of heading
  # are detected correct.
  if np.sum(flip_scalar) < 0:
    flip_scalar *= -1
  need_flip = flip_scalar < 0
  # Flip the heading and speed, -PI < heading < PI
  xs[need_flip, 2] -= np.pi
  xs[xs[:,2] < -np.pi, 2] += np.pi
  xs[need_flip, 3] *= -1
  us[need_flip, 0] *= -1


# wrap to [-pi, pi]
def angle_wrap(theta):
  return (theta + np.pi) % (2 * np.pi) - np.pi


def cost(xs_in, us_in, x, u, t_i):
  Q = np.diag([10, 10, 20/3, 10/2])
  R = 0.1 * np.diag([1, 1])
  assert x.shape[-1] == Q.shape[0]
  assert u.shape[-1] == R.shape[0]

  x_diff = x - xs_in[t_i]
  # wrap heading to [-pi, pi]
  x_diff = jnp.asarray([angle_wrap(x_diff[i]) if i == 2 else x_diff[i] for i in range(4)])
  u_diff = u - us_in[t_i]

  return x_diff.T @ Q @ x_diff + u_diff.T @ R @ u_diff


def dynamics(dts, x, u, t_i):
  pos_x = x[0]  # pos_x
  pos_y = x[1]  # pos_y
  heading = x[2]  # heading
  v = x[3]  # velocity

  a = u[0]  # acceleration
  omega = u[1]  # angular_velocity

  # A = [[1, 0, 0, dt * cos(heading)],
  #      [0, 1, 0, dt * sin(heading)],
  #      [0, 0, 1, 0]
  #      [0, 0, 0, 1]]
  # B = [[0, 0],
  #      [0, 0],
  #      [0, dt],
  #      [dt, 0]]
  next_pos_x = pos_x + dts[t_i] * jnp.cos(heading) * v
  next_pos_y = pos_y + dts[t_i] * jnp.sin(heading) * v
  next_heading = heading + dts[t_i] * omega
  next_v = v + dts[t_i] * a

  return jnp.array([next_pos_x, next_pos_y, next_heading, next_v])


def plot_result(xs_in, us_in, xs_out, us_out, dts):
  xs_calculated_by_us = [[xs_in[0,0], xs_in[0,1]]]
  for i in range(len(us_in)):
    pos_x = xs_calculated_by_us[i][0]
    pos_y = xs_calculated_by_us[i][1]
    heading = xs_in[i, 2]
    v = xs_in[i, 3]

    next_pos_x = pos_x + dts[i] * np.cos(heading) * v
    next_pos_y = pos_y + dts[i] * np.sin(heading) * v
    xs_calculated_by_us.append([next_pos_x, next_pos_y])
  xs_calculated_by_us = np.array(xs_calculated_by_us)

  v1 = []
  a1 = []
  h1 = []
  o1 = []
  for i in range(len(us_in)):
    v1.append(np.linalg.norm(xs_in[i+1, 0:2] - xs_in[i, 0:2]) / dts[i])
    a1.append((xs_in[i+1, 3] - xs_in[i, 3]) / dts[i])
    h1.append(np.arctan2(xs_in[i+1, 1] - xs_in[i, 1], xs_in[i+1, 0] - xs_in[i, 0]))
    o1.append((xs_in[i+1, 2] - xs_in[i, 2]) / dts[i])
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

  xs_in = jnp.asarray(xs)
  us_in = jnp.asarray(us[:-1])
  dts = jnp.ones(len(xs)) * 0.1  # Discrete time-steps in seconds
  # iLQR optimization
  xs_out, us_out, _, _, _, _, _ = optimizers.ilqr(
      cost=partial(cost, xs_in, us_in),
      dynamics=partial(dynamics, dts),
      x0=xs_in[0],
      U=us_in)

  plot_result(xs, us[:-1], xs_out, us_out, dts)
  print('Done')
