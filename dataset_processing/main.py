import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

images_dir  = "images/"
masks_dir   = "masks/"
out_dir     = "out/"
train_dir   = out_dir + "train/"
val_dir     = out_dir + "val/"
count       = 20
result_size = 64
val_number  = 3
result_size_arr = (result_size, result_size, result_size)

images = [np.load(images_dir + str(i) + ".npy") for i in range(count)]
masks  = [np.load(masks_dir + str(i) + ".npy") for i in range(count)]

print(np.array(images).shape, np.array(masks).shape)

images = [np.squeeze(image, 3) for image in images]
masks = [np.squeeze(mask, 3) for mask in masks]

print(np.array(images).shape, np.array(masks).shape)

images = [resize(i, result_size_arr, order=0, anti_aliasing=False) for i in images]
masks = [resize(i, result_size_arr, order=0, anti_aliasing=False) for i in masks]

print(np.array(images).shape, np.array(masks).shape)

for i in range(count):
    dir = train_dir if i < count - val_number else val_dir
    name = "pt" + str(i) + "_data" + ".npz"
    np.savez(dir + name, arr_0=images[i], arr_1=masks[i])

'''
for i in range(result_size):
    plt.subplot(121)
    plt.imshow(images[0][i])
    plt.subplot(122)
    plt.imshow(masks[0][i])
    plt.show()
'''