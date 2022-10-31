import json

from ilqr.cost import QRCost
from ilqr.dynamics import BatchAutoDiffDynamics
from ilqr import iLQR
from matplotlib import pyplot as plt
import numpy as np
import theano.tensor as T

STATE_SIZE = 3  # [position_x, position_y, heading]
ACTION_SIZE = 2  # [velocity, angular_velocity]

dt = 0.1  # Discrete time-step in seconds

# global source trajectory data, used in function f.
xs = None  # State data of the trajectory
us = None  # Control state of the trajectory

def f(x, u, i):
  """Batched implementation of the dynamic model.

  Args:
    x: State vector [delta(position_x), delta(position_y), delta(heading)].
    u: Control vector [delta(velocity), delta(angular_velocity)].
    i: Current time index.

  Returns:
    Next state vector.
  """

  d_pos_x = x[..., 0]
  d_pos_y = x[..., 1]
  d_heading = x[..., 2]
  d_v = u[..., 0]
  d_omega = u[..., 1]

  # Not batched yet, and I have no idea how to query index data from a theano
  # tensor variable i.
  heading = xs[i, 2]
  v = us[i, 0]

  # I'm not sure if this is correct, since omega is not independent with v.
  # A = [[1, 0, -dt * v * sin(heading)],
  #      [0, 1, dt * v * cos(heading)],
  #      [0, 0, 1]]
  # B = [[dt * cos(heading), 0],
  #      [dt * sin(heading), 0],
  #      [0, dt]]

  sh_dt = T.sin(heading) * dt
  ch_dt = T.cos(heading) * dt

  # A*dx + B*du
  next_d_pos_x = d_pos_x - v * sh_dt * d_heading + ch_dt * d_v
  next_d_pos_y = d_pos_y + v * ch_dt * d_heading + sh_dt * d_v
  next_d_heading = d_heading + dt * d_omega

  return T.stack([
    next_d_pos_x,
    next_d_pos_y,
    next_d_heading
  ]).T


def load_trajectory_data(filepath):
  global xs
  global us

  with open(filepath) as f:
    data = json.load(f)

    xs = []
    us = []
    for d in data['trajectoryData']:
      xs.append([d['xPos'], d['yPos'], d['heading']])
      us.append([d['speed'], d['omega']])
  xs = np.array(xs)
  us = np.array(us)[:-1]


if __name__ == '__main__':
  load_trajectory_data('a_noisy_trajectory.json')

  # Compile the dynamics.
  dynamics = BatchAutoDiffDynamics(f, STATE_SIZE, ACTION_SIZE)

  # Cost function
  Q = 10 * np.eye(STATE_SIZE)
  R = 0.1 * np.eye(ACTION_SIZE)
  cost = QRCost(Q, R)

  ilqr = iLQR(dynamics, cost, len(us))
  d_xs_out, d_us_out = ilqr.fit(np.zeros(STATE_SIZE), np.zeros_like(us))

  plt.figure()
  plt.plot(xs[:,0], xs[:,1], 'r')
  plt.plot(d_xs_out[:,0] + xs[:,0], d_xs_out[:,1] + xs[:,1], 'b')
  plt.show()

  print('Done')
