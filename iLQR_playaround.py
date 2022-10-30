from ilqr.cost import QRCost
from ilqr.dynamics import BatchAutoDiffDynamics
from ilqr import iLQR
import numpy as np
import theano.tensor as T

STATE_SIZE = 3  # [position_x, position_y, heading]
ACTION_SIZE = 2  # [velocity, angular_velocity]

N = 1000  # Number of time-steps in trajectory.
dt = 0.01  # Discrete time-step in seconds
L = 2  # Wheelbase of the agent

def f(x, u, i):
  """Batched implementation of the dynamic model.

  Args:
    x: State vector [position_x, position_y, heading].
    u: Control vector [velocity, angular_velocity].
    i: Current time index.

  Returns:
    Next state vector.
  """

  pos_x = x[..., 0]
  pos_y = x[..., 1]
  heading = x[..., 2]
  v = u[..., 0]
  omega = u[..., 1]

  heading_i = 0
  v_i = 0
  omega_i = 0

  sh_dt = T.sin(heading_i) * dt
  ch_dt = dt * T.cos(heading_i) * dt

  # A = [[1, 0, dt * v_i * sin(heading_i)],
  #      [0, 1, dt * v_i * cos(heading_i)],
  #      [0, 0, 1]]
  # B = [[dt * cos(heading_i), 0],
  #      [dt * sin(heading_i), 0],
  #      [dt * tan(omega_i) / L, v_i * dt / (L * cos(omega_i) ** 2)]]

  # Ax + Bu
  next_pos_x = pos_x + v_i * sh_dt * heading + ch_dt * v
  next_pos_y = pos_y + v_i * ch_dt * heading + sh_dt * v
  next_heading = (heading + dt * T.tan(omega_i) / L * v +
                  v_i * dt / (L * T.cos(omega_i) ** 2) * omega)

  return T.stack([
    next_pos_x,
    next_pos_y,
    next_heading
  ]).T

if __name__ == '__main__':
  # Compile the dynamics.
  dynamics = BatchAutoDiffDynamics(f, STATE_SIZE, ACTION_SIZE)

  # Cost function
  Q = 10 * np.eye(STATE_SIZE)
  R = 0.1 * np.eye(ACTION_SIZE)
  cost = QRCost(Q, R)


  # Test
  initial_x = np.array([2, 1, np.pi / 4])
  initial_u = np.array([1, np.pi / 8])
  us_init = np.random.uniform(-1, 1, (N, 2)) # Random initial action path.
  i = 0

  # print(dynamics.f(initial_x, initial_u, i))
  # print(dynamics.f_x(initial_x, initial_u, i))
  # print(dynamics.f_u(initial_x, initial_u, i))

  ilqr = iLQR(dynamics, cost, N)
  xs, us = ilqr.fit(initial_x, us_init)

  print('Done')
