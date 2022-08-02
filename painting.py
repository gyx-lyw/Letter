from matplotlib import pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(1, 1)

f1 = open(f'letter_1.txt')
lines = f1.readlines()
q = []
w = []
for line in lines:
    unit = line.split(',')
    it = float(unit[0].split(':')[1])
    loss = float(unit[1].split(':')[1])
    q.append(it)
    w.append(loss)
q = np.array(q)
w = np.array(w)
ax1.plot(q, w, c='r', label='learning_rate=0.001,direction_num=1')
f1.close()


f1 = open(f'letter_5.txt')
lines = f1.readlines()
a = []
b = []
for line in lines:
    unit = line.split(',')
    it = float(unit[0].split(':')[1])
    loss = float(unit[1].split(':')[1])
    a.append(it)
    b.append(loss)
a = np.array(a)
b = np.array(b)
ax1.plot(a, b, c='g', label='learning_rate=0.002,direction_num=5')
f1.close()


f1 = open(f'letter_10.txt')
lines = f1.readlines()
a = []
b = []
for line in lines:
    unit = line.split(',')
    it = float(unit[0].split(':')[1])
    loss = float(unit[1].split(':')[1])
    a.append(it)
    b.append(loss)
a = np.array(a)
b = np.array(b)
ax1.plot(a, b, c='b', label='learning_rate=0.0025,direction_num=10')
f1.close()


f1 = open(f'experiment_1.txt')
lines = f1.readlines()
a = []
b = []
for line in lines:
    unit = line.split(',')
    it = float(unit[0].split(':')[1])
    loss = float(unit[1].split(':')[1])
    a.append(it)
    b.append(loss)
a = np.array(a)
b = np.array(b)
ax1.plot(a, b, c='orange', label='learning_rate=0.0002,bp')
f1.close()


ax1.set_ylim(0, 4)
xtick = range(0, 200, 10)
ax1.legend()
plt.xticks(xtick)
plt.xlabel('Iterations')
ax1.set_ylabel('Loss')
plt.show()
