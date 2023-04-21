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


# wrap to [-pi, pi]
def angle_wrap(theta):
  return (theta + np.pi) % (2 * np.pi) - np.pi


# Since heading data maybe detected flipped, we did a preprocess here to flip
# the heading by assuming heading should be continuous and more than 50% of
# heading are detected correct.
def preprocess_flip_heading(xs, us):
  # Heading direction from input data
  data_dirs = np.c_[np.cos(xs[:, 2]), np.sin(xs[:, 2])]
  # Indices that break the continuous assumption (by checking the inner product
  # of adjacent heading directions)
  data_discontinuous = np.sum(data_dirs[:-1] * data_dirs[1:], axis=1) < 0
  adj_flip_scalar = np.ones(len(xs))
  adj_flip_scalar[1:][data_discontinuous] *= -1
  flip_scalar = np.cumprod(adj_flip_scalar)
  # Reverse flip_scalar based on the 2nd assumption: more than 50% of heading
  # are detected correct
  if np.sum(flip_scalar) < 0:
    flip_scalar *= -1
  need_flip = flip_scalar < 0
  # Flip the heading, speed and acceleration
  xs[need_flip, 2] = angle_wrap(xs[need_flip, 2] - np.pi)
  xs[need_flip, 3] *= -1
  us[need_flip, 0] *= -1


def cost(Q, R, xs_in, us_in, x, u, t_i):
  x_diff = x - xs_in[t_i]
  # wrap heading to [-pi, pi]
  x_diff = x_diff.at[2].set(angle_wrap(x_diff[2]))
  u_diff = u - us_in[t_i]

  return x_diff.T @ Q @ x_diff + u_diff.T @ R @ u_diff


def dynamics(dts, x, u, t_i):
  # _ = x[0]  # pos_x
  # _ = x[1]  # pos_y
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
  # x_next = A * x + B * u
  return x + jnp.array(
      [jnp.cos(heading) * v, jnp.sin(heading) * v, omega, a]) * dts[t_i]


def plot_result(xs_in, us_in, xs_out, us_out, dts):
  N = len(us_in)

  # pos calculated by pos[0] and origin control
  xs_ref = [[xs_in[0,0], xs_in[0,1]]]
  for i in range(N):
    pos_x = xs_ref[i][0]
    pos_y = xs_ref[i][1]
    heading = xs_in[i, 2]
    v = xs_in[i, 3]

    next_pos_x = pos_x + dts[i] * np.cos(heading) * v
    next_pos_y = pos_y + dts[i] * np.sin(heading) * v
    xs_ref.append([next_pos_x, next_pos_y])
  xs_ref = np.array(xs_ref)

  # values from derivative
  v_deriv = np.linalg.norm(xs_in[1:, 0:2] - xs_in[:-1, 0:2], axis=1) / dts
  a_deriv = (xs_in[1:, 3] - xs_in[:-1, 3]) / dts
  h_deriv = (
      np.arctan2(xs_in[1:, 1] - xs_in[:-1, 1], xs_in[1:, 0] - xs_in[:-1, 0]))
  o_deriv = (xs_in[1:, 2] - xs_in[:-1, 2]) / dts

  # flip based on the heading data
  data_dirs = np.c_[np.cos(h_deriv), np.sin(h_deriv)]
  discontinuous_flag = np.sum(data_dirs[:-1] * data_dirs[1:], axis=1) < 0
  adj_flip_scalar = np.ones(N)
  adj_flip_scalar[1:][discontinuous_flag] *= -1
  flip_scalar = np.cumprod(adj_flip_scalar)
  if np.sum(flip_scalar) < 0:
    flip_scalar *= -1
  need_flip = flip_scalar < 0
  h_deriv[need_flip] = angle_wrap(h_deriv[need_flip] - np.pi)
  v_deriv[need_flip] *= -1
  a_deriv[need_flip] *= -1

  from matplotlib import pyplot as plt
  plt.figure()
  plt.subplot(2,3,1)
  plt.title('origin pos, origin integrated, optimization result')
  plt.plot(xs_in[:,0], xs_in[:,1], 'r.-', label='origin pos(m)')
  plt.plot(xs_out[:,0], xs_out[:,1], 'g.-', label='output pos')
  plt.plot(xs_ref[:,0], xs_ref[:,1], 'y.-',
           label='pos calculated from pos[0] and origin control')
  plt.axis('equal')
  plt.legend()

  plt.subplot(2,3,2)
  plt.title('speed')
  plt.plot(np.arange(N+1), xs_in[:,3], 'r', label='origin speed(m/s)')
  plt.plot(np.arange(N+1), xs_out[:,3], 'g', label='output speed')
  plt.plot(np.arange(N), v_deriv, 'r:', label='derivative of origin pos')
  plt.legend()

  plt.subplot(2,3,3)
  plt.title('accel')
  plt.plot(np.arange(N), us_in[:,0], 'r', label='origin accel(m/s2)')
  plt.plot(np.arange(N), us_out[:,0], 'g', label='output accel')
  plt.plot(np.arange(N), a_deriv, 'r:', label='derivative of origin speed')
  plt.legend()

  plt.subplot(2,3,4)
  plt.title('heading, [-pi, pi]')
  plt.plot(np.arange(N+1), xs_in[:,2], 'r', label='origin heading(rad)')
  plt.plot(np.arange(N+1), xs_out[:,2], 'g', label='output heading')
  plt.plot(np.arange(N), h_deriv, 'r:', label='derivative of origin pos')
  plt.legend()
  plt.ylim(-np.pi, np.pi)

  plt.subplot(2,3,5)
  plt.title('omega')
  plt.plot(np.arange(N), us_in[:,1], 'r', label='origin omega(rad/s)')
  plt.plot(np.arange(N), us_out[:,1], 'g', label='output omega')
  plt.plot(np.arange(N), o_deriv, 'r:', label='derivative of origin heading')
  plt.legend()
  plt.ylim(-np.pi, np.pi)
  plt.show()


if __name__ == '__main__':
  xs, us = load_trajectory_data('a_noisy_trajectory.json')
  assert len(xs) > 1
  preprocess_flip_heading(xs, us)

  xs_in = jnp.asarray(xs)
  us_in = jnp.asarray(us[:-1])
  # Potential varying time-steps in seconds
  dts = jnp.ones(len(us_in)) * 0.1  # Discrete time-steps in seconds
  # Cost weights
  Q = np.diag([10, 10, 20/3, 10/2])
  R = 0.1 * np.diag([1, 1])
  assert xs.shape[-1] == Q.shape[0]
  assert us.shape[-1] == R.shape[0]
  # iLQR optimization
  xs_out, us_out, _, _, _, _, _ = optimizers.ilqr(
      cost=partial(cost, Q, R, xs_in, us_in),
      dynamics=partial(dynamics, dts),
      x0=xs_in[0],
      U=us_in)

  plot_result(xs, us[:-1], xs_out, us_out, np.asarray(dts))
  print('Done')
