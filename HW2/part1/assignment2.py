import matplotlib.pyplot as plt
import numpy as np

def plot_patterns(k, ax):
    #ax = fig.add_subplot(projection='3d')
    xs = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    ys = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    zs = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    # set colors
    colors = np.array(['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'], order = 'C')
    for it in range(k):
        colors[it] = 'r'
    # Plot +1
    ax.scatter(xs[:k], ys[:k], zs[:k],c = colors[:k], s= 100, marker='s')
    # Plot -1
    ax.scatter(xs[k:], ys[k:], zs[k:],c = colors[k:], s= 100, marker='s')
    #ax.legend(['1','0'])
    #ax.set_xlabel('x_1')
    #ax.set_ylabel('x_2')
    #ax.set_zlabel('x_3')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])
    ax.set_xticklabels(['0','1'])
    ax.set_yticklabels(['0','1'])
    ax.set_zticklabels(['0','1'])
    return ax

def plot_patterns4(ax):
    k = 4
    xs = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    ys = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    zs = np.array([0, 0, 1, 0, 1, 0, 1, 1])
    # set colors
    colors = np.array(['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'], order = 'C')
    for it in range(k):
        colors[it] = 'r'
    # Plot +1
    ax.scatter(xs[:k], ys[:k], zs[:k],c = colors[:k], s= 100, marker='s')
    # Plot -1
    ax.scatter(xs[k:], ys[k:], zs[k:],c = colors[k:], s= 100, marker='s')
    #ax.legend(['1','0'])
    #ax.set_xlabel('x_1')
    #ax.set_ylabel('x_2')
    #ax.set_zlabel('x_3')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])
    ax.set_xticklabels(['0','1'])
    ax.set_yticklabels(['0','1'])
    ax.set_zticklabels(['0','1'])
    return ax

def plot_insep1(ax):
    k = 2
    xs = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    ys = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    zs = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    # set colors
    colors = np.array(['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'], order = 'C')
    for it in range(k):
        colors[it] = 'r'
    # Plot +1
    ax.scatter(xs[:k], ys[:k], zs[:k],c = colors[:k], s= 100, marker='s')
    # Plot -1
    ax.scatter(xs[k:], ys[k:], zs[k:],c = colors[k:], s= 100, marker='s')
    #ax.legend(['1','0'])
    #ax.set_xlabel('x_1')
    #ax.set_ylabel('x_2')
    #ax.set_zlabel('x_3')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])
    ax.set_xticklabels(['0','1'])
    ax.set_yticklabels(['0','1'])
    ax.set_zticklabels(['0','1'])
    return ax

def plot_insep2(ax):
    k = 4
    xs = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    ys = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    zs = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    # set colors
    colors = np.array(['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'], order = 'C')
    for it in range(k):
        colors[it] = 'r'
    # Plot +1
    ax.scatter(xs[:k], ys[:k], zs[:k],c = colors[:k], s= 100, marker='s')
    # Plot -1
    ax.scatter(xs[k:], ys[k:], zs[k:],c = colors[k:], s= 100, marker='s')
    #ax.legend(['1','0'])
    #ax.set_xlabel('x_1')
    #ax.set_ylabel('x_2')
    #ax.set_zlabel('x_3')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])
    ax.set_xticklabels(['0','1'])
    ax.set_yticklabels(['0','1'])
    ax.set_zticklabels(['0','1'])
    return ax

fig = plt.figure(figsize = plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1, projection = '3d')
plot_insep1(ax)
ax.set_title('XOR, XNOR')
ax = fig.add_subplot(1,2,2, projection = '3d')
plot_insep2(ax)
ax.set_title('Opposite corner problem')
plt.savefig('assignment2_2.jpg')

fig = plt.figure(figsize = plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1, projection = '3d')
plot_patterns(0,ax)
ax.set_title('k=0')
ax = fig.add_subplot(2,3,2, projection = '3d')
plot_patterns(1,ax)
ax.set_title('k=1')
ax = fig.add_subplot(2,3,3, projection = '3d')
plot_patterns(2,ax)
ax.set_title('k=2')
ax = fig.add_subplot(2,3,4, projection = '3d')
plot_patterns(3,ax)
ax.set_title('k=3')
ax = fig.add_subplot(2,3,5, projection = '3d')
plot_patterns(4,ax)
ax.set_title('k=4')
ax = fig.add_subplot(2,3,6, projection = '3d')
plot_patterns4(ax)
ax.set_title('k=4')
plt.savefig('assignment2.jpg')





