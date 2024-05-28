import numpy as np
import gzip
from datetime import datetime
import matplotlib.pyplot as plt
import barycenter
import time
import pickle


def pic_to_pointlist(input:np.ndarray, repeat = 1, z = 0, z_num = 0, r = 10000):
    output = []
    weight = []
    input = input.reshape((28,28))
    for i in range(28):
        for j in range(28):
            if input[i,j] > 10:
                output.append([i, j])
                weight.append(input[i, j])
    output = np.array(output).repeat(repeat, 0)
    weight = np.array(weight).repeat(repeat, 0)
    weight = weight / sum(weight) * (1 - z)
    if z:
        theta = np.random.uniform(0, 2 * np.pi, z_num)
        # 计算点的坐标
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        output = np.concatenate([output, np.column_stack((x,y))])
        weight = np.concatenate([weight, np.full(z_num, z / z_num)])
    return output, weight / sum(weight)



data_file = 'data/train-images-idx3-ubyte.gz'
with gzip.open(data_file, 'rb') as f:
    file_content = f.read()

data = np.frombuffer(file_content, dtype=np.uint8, offset=16).reshape((60000, 784)).astype(np.float64)

label_file = 'data/train-labels-idx1-ubyte.gz'

with gzip.open(label_file, 'rb') as f:
    file_content = f.read()

label = np.frombuffer(file_content, dtype=np.uint8, offset=8)

shuffle = np.array(range(60000))

pic_num = 5
point_lists = np.array([(i, j) for j in range(28) for i in range(28)])[np.newaxis, :].repeat(pic_num, axis=0)

# barycenter = np.array([(i, j) for j in range(0, 28, 2) for i in range(0, 28, 2)])
w_list = np.ones((pic_num,)) / pic_num

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
# plt.suptitle('Robust Wasserstein Barycenters for MNIST')
cm = 'Blues'

ans = []
# ans = pickle.load(open('/data/yqy/nips/gurobi/2024:01:04-00:22:29mnist.pkl', 'rb'))

np.random.seed(int(time.time()))

size = 40
plt.clf()
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    u_list = []
    point_lists = []
    np.random.shuffle(shuffle)
    for j in range(60000):
        if label[shuffle[j]] == i:
            point_list, weight = pic_to_pointlist(data[shuffle[j]])
            u_list.append(weight)
            point_lists.append(point_list)
        if len(u_list) == pic_num:
            break
    b_weight, _, _, _, center = barycenter.barycenter(point_lists, u_list, w_list, 0, 0, size, 0, div=2)
    ans.append([b_weight, center])
    axes[i // 5, i % 5].scatter(center[:,1], -center[:,0], label='Scatter Plot', cmap='Blues', c=b_weight, s=b_weight*4000)
    axes[i // 5, i % 5].set_xlim(0, 28)
    axes[i // 5, i % 5].set_ylim(-28, 0)
    axes[i // 5, i % 5].axis('off')
    print(i, end='\r')


f = open('output/' + datetime.now().strftime("%Y:%m:%d-%H:%M:%S") + f'_{size}mnist.pkl', 'wb')
pickle.dump(ans, f, -1)
f.close()
plt.tight_layout()
plt.savefig('output/' + datetime.now().strftime("%Y:%m:%d-%H:%M:%S") + f'_{size}mnist.png', dpi=600, transparent=False)