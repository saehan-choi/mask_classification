import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import numpy as np


fig = plt.figure(figsize=(9,6))
ax  = fig.add_subplot(111, projection = '3d')

def read_line(output):
    output = output.readline().split('/')

    return output

mask = open('./output_labels/mask_label.txt', 'r')
mask = read_line(mask)

nomask = open('./output_labels/nomask_label.txt', 'r')
nomask = read_line(nomask)

wrong = open('./output_labels/wrong_label.txt', 'r')
wrong = read_line(wrong)

x_mask = []
y_mask = []
z_mask = []

x_nomask = []
y_nomask = []
z_nomask = []

x_wrong = []
y_wrong = []
z_wrong = []
try:
    for i in range(len(mask)):
        line = mask[i]
        line = line[1:-1].split(',')

        line[0] = float(line[0])
        line[1] = float(line[1])
        line[2] = float(line[2])

        x_mask.append(line[0])
        y_mask.append(line[1])
        z_mask.append(line[2])
except:
    pass

try:
    for i in range(len(nomask)):

        line = nomask[i]
        line = line[1:-1].split(',')

        line[0] = float(line[0])
        line[1] = float(line[1])
        line[2] = float(line[2])

        x_nomask.append(line[0])
        y_nomask.append(line[1])
        z_nomask.append(line[2])
except:
    pass

try:
    for i in range(len(wrong)):

        line = wrong[i]
        line = line[1:-1].split(',')

        line[0] = float(line[0])
        line[1] = float(line[1])
        line[2] = float(line[2])

        x_wrong.append(line[0])
        y_wrong.append(line[1])
        z_wrong.append(line[2])

except:
    pass


ax.scatter(x_mask, y_mask, z_mask, color='r', alpha=0.5, label='mask')
ax.scatter(x_nomask, y_nomask, z_nomask, color='g', alpha=0.5, label='nomask')
ax.scatter(x_wrong, y_wrong, z_wrong, color='b', alpha=0.5, label='wrong')

ax.set_xlabel('x axis similarity')
ax.set_ylabel('y axis similarity')
ax.set_zlabel('z axis similarity')

plt.legend()
plt.title('efficientnet-b0 prediction visualization')
plt.show()

