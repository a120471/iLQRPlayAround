import json

from ilqr.cost import PathQRCost
from ilqr.dynamics import BatchAutoDiffDynamics
from ilqr import iLQR
from matplotlib import pyplot as plt
import numpy as np
import theano.tensor as T

STATE_SIZE = 4  # [position_x, position_y, heading, velocity]
ACTION_SIZE = 2  # [acceleration, angular_velocity]

dt = 0.1  # Discrete time-step in seconds

# global source trajectory data, used in function f.
xs = None  # State data of the trajectory
us = None  # Control state of the trajectory

def load_trajectory_data(filepath):
  global xs
  global us

  with open(filepath) as f:
    data = json.load(f)

    xs = []
    us = []
    for d in data['trajectoryData']:
      xs.append([d['xPos'], d['yPos'], d['heading'], d['speed']])
      us.append([d['accel'], d['omega']])
  xs = np.array(xs)
  us = np.array(us)


# Heading data maybe detected flipped, and the input data assumed no reverse
# driving. We flip the heading by assuming heading should be continuous and more
# than 50% of heading are detected correct.
def preprocess_flip_heading():
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


def f(x, u, i):
  """Batched implementation of the dynamic model.

  Args:
    x: State vector [position_x, position_y, heading, velocity].
    u: Control vector [acceleration, angular_velocity].
    i: Current time index.

  Returns:
    Next state vector.
  """

  pos_x = x[..., 0]
  pos_y = x[..., 1]
  heading = x[..., 2]
  v = x[..., 3]
  a = u[..., 0]
  omega = u[..., 1]

  # A = [[1, 0, 0, dt * cos(heading)],
  #      [0, 1, 0, dt * sin(heading)],
  #      [0, 0, 1, 0]
  #      [0, 0, 0, 1]]
  # B = [[0, 0],
  #      [0, 0],
  #      [0, dt],
  #      [dt, 0]]

  # Ax + Bu (non linear)
  next_pos_x = pos_x + dt * T.cos(heading) * v
  next_pos_y = pos_y + dt * T.sin(heading) * v
  next_heading = heading + dt * omega
  next_v = v + dt * a

  return T.stack([
    next_pos_x,
    next_pos_y,
    next_heading,
    next_v
  ]).T


def f_inv(x, u, i):
  pos_x = x[..., 0]
  pos_y = x[..., 1]
  heading = x[..., 2]
  v = x[..., 3]
  a = u[..., 0]
  omega = u[..., 1]

  # A = [[1, 0, 0, -dt * cos(heading)],
  #      [0, 1, 0, -dt * sin(heading)],
  #      [0, 0, 1, 0]
  #      [0, 0, 0, 1]]
  # B = [[0, 0],
  #      [0, 0],
  #      [0, -dt],
  #      [-dt, 0]]

  # Ax + Bu (non linear)
  next_pos_x = pos_x - dt * T.cos(heading) * v
  next_pos_y = pos_y - dt * T.sin(heading) * v
  next_heading = heading - dt * omega
  next_v = v - dt * a

  return T.stack([
    next_pos_x,
    next_pos_y,
    next_heading,
    next_v
  ]).T


def plot_result(xs_in, us_in, xs_out, us_out):
  xs_calculated_by_us = [[xs_in[0,0], xs_in[0,1]]]
  for i in range(len(us_in)):
    pos_x = xs_calculated_by_us[i][0]
    pos_y = xs_calculated_by_us[i][1]
    heading = xs_in[i, 2]
    v = xs_in[i, 3]

    next_pos_x = pos_x + dt * np.cos(heading) * v
    next_pos_y = pos_y + dt * np.sin(heading) * v
    xs_calculated_by_us.append([next_pos_x, next_pos_y])
  xs_calculated_by_us = np.array(xs_calculated_by_us)

  plt.figure()
  plt.subplot(2,3,1)
  plt.title('origin pos, origin_integrated, result')
  plt.plot(xs_in[:,0], xs_in[:,1], 'r')
  plt.plot(xs_out[:,0], xs_out[:,1], 'b')
  plt.plot(xs_calculated_by_us[:,0], xs_calculated_by_us[:,1], 'g')
  plt.subplot(2,3,2)
  plt.title('heading')
  plt.plot(np.arange(len(xs_in[:,2])), xs_in[:,2], 'r')
  plt.plot(np.arange(len(xs_in[:,2])), xs_out[:,2], 'b')
  plt.ylim(-np.pi, np.pi)
  plt.subplot(2,3,3)
  plt.title('speed')
  plt.plot(np.arange(len(xs_in[:,3])), xs_in[:,3], 'r')
  plt.plot(np.arange(len(xs_in[:,3])), xs_out[:,3], 'b')
  plt.subplot(2,3,4)
  plt.title('omega')
  plt.plot(np.arange(len(us_in[:,1])), us_in[:,1], 'r')
  plt.plot(np.arange(len(us_in[:,1])), us_out[:,1], 'b')
  plt.ylim(-np.pi, np.pi)
  plt.subplot(2,3,5)
  plt.title('accel')
  plt.plot(np.arange(len(us_in[:,0])), us_in[:,0], 'r')
  plt.plot(np.arange(len(us_in[:,0])), us_out[:,0], 'b')
  plt.show()

if __name__ == '__main__':
  load_trajectory_data('a_noisy_trajectory.json')
  preprocess_flip_heading()

  # Forward optimization
  xs_in = xs
  us_in = us[:-1]
  # Compile the dynamics.
  dynamics = BatchAutoDiffDynamics(f, STATE_SIZE, ACTION_SIZE)
  # Cost function
  Q = 1 * np.diag([1, 1, 2/3, 1/2])
  R = 0.1 * np.eye(ACTION_SIZE)
  cost = PathQRCost(Q, R, xs_in, us_in)
  # ilqr optimization
  ilqr = iLQR(dynamics, cost, len(us_in))
  xs_out, us_out = ilqr.fit(xs_in[0], us_in)
  us_out_backup = np.copy(us_out)

  # In order to optimize the beginning data, we also perform a backward
  # optimization.
  xs_in = xs_out[::-1]
  us_in = us[-1:0:-1]
  # Compile the dynamics.
  dynamics = BatchAutoDiffDynamics(f_inv, STATE_SIZE, ACTION_SIZE)
  # Cost function
  cost = PathQRCost(Q, R, xs_in, us_in)
  # ilqr optimization
  ilqr = iLQR(dynamics, cost, len(us_in))
  xs_out, us_out = ilqr.fit(xs_in[0], us_in)
  xs_out = xs_out[::-1]
  us_out = np.r_[us_out_backup[:1], us_out][::-1]

  plot_result(xs, us[:-1], xs_out, us_out[:-1])
  print('Done')
