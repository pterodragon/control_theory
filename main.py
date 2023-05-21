from kalman_filter import KalmanFilter
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from collections import deque
import numpy as np

MAX_X = 200
MAX_Y = 6
measurement_line = deque([0.0] * MAX_X, maxlen=MAX_X)
prediction_line = deque([0.0] * MAX_X, maxlen=MAX_X)
ground_truth_line = deque([0.0] * MAX_X, maxlen=MAX_X)

t = 0
dt = np.pi / 60

# dynamics for the ODE y'' + y' = 0; y(0) = 0; y'(0) = 2; solution is y = 2 * sin(x)
F = np.array([[1, dt, 0], [0, 1, dt], [0, -1, 0]])
H = np.array([1, 0, 0]).reshape(1, 3)
Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
R = np.array([0.5]).reshape(1, 1)

x0 = np.array([0, 2, -2]).T

kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0)

up = np.array([0.1,0,0]).T
down = np.array([-0.1,0,0]).T
acc = np.zeros(3).T
state = 0

def update(fn, l1, l2, l3):
    global state, acc
    u = np.zeros(3).T
    if state == 1:
        u = up
        acc += u
    elif state == 2:
        u = down
        acc += u
    state = 0


    global t
    t += dt
    ground_truth = 2 * np.sin(t) + acc[0]
    measurement = 2 * np.sin(t) + acc[0] + np.random.normal(0, 1)
    ground_truth_line.append(ground_truth)
    prediction_line.append(kf.predict(u)[0])
    kf.update(measurement)
    measurement_line.append(measurement)
    l1.set_data(range(-MAX_X // 2, MAX_X // 2), ground_truth_line)
    l2.set_data(range(-MAX_X // 2, MAX_X // 2), prediction_line)
    l3.set_data(range(-MAX_X // 2, MAX_X // 2), measurement_line)

fig = plt.figure()

a = plt.axes(xlim=(-(MAX_X // 2), MAX_X // 2), ylim=(-(MAX_Y // 2), MAX_Y // 2))
(l1,) = a.plot([], [], label="Ground Truth")
(l2,) = a.plot([], [], label="KF Prediction")
(l3,) = a.plot([], [], label="Measurement")
ani = anim.FuncAnimation(fig, update, fargs=(l1,l2,l3), interval=50)
def on_press(event):
    global state
    if event.key == 'up':
        state = 1
    elif event.key == 'down':
        state = 2
    elif event.key == 'x':
        visible = l1.get_visible()
        l1.set_visible(not visible)
        fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_press)

plt.legend(loc='upper left')
plt.show()

# input u: TODO
