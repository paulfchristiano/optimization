import numpy as np
import time
import array
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#UTILITIES

def inner(f, image1, image2):
    h = f(np.array([image1, image2]))
    return np.dot(h[0], h[1])

def training_data(images):
    return ( [image[0] for image in images], [image[1] for image in images] )

def read_images(mode='train', n=None):
    image_file, label_file = '', ''
    if mode == 'train':
        image_file="train-images-idx3-ubyte"
        label_file="train-labels-idx1-ubyte"
    elif mode == 'test':
        image_file="t10k-images-idx3-ubyte"
        label_file="t10k-labels-idx1-ubyte"
    with open(image_file, 'r') as f:
        f.read(4)
        num_images = parse_int(f.read(4))
        if n and n < num_images:
            num_images = n
        image_height = parse_int(f.read(4))
        image_width = parse_int(f.read(4))
        image_size = image_height * image_width
        images = [ np.array([ord(f.read(1)) / 255.0 for y in range(image_size)] ) for i in range(num_images) ]
    with open(label_file, 'r') as f:
        f.read(4)
        num_images = parse_int(f.read(4))
        if n and n < num_images:
            num_images = n
        labels = [ ord(f.read(1)) for i in range(num_images) ]
    return (images, labels)

def show(images, width=28):
    if len(images) == width**2:
        return show([images])
    if len(images) == 1:
        return show(images + images)
    fig, plots = plt.subplots(1, len(images))
    for i in range(len(images)):
        image = images[i]
        if len(np.shape(image)) == 1:
            height = len(image) / width
            plots[i].imshow([ [ 1 - image[i*width + j] for j in range(width)] for i in range(height) ], cmap = cm.Greys_r)
        else:
            plots[i].imshow(image, cmap=cm.Greys_r)
    plt.show()

def parse_int(b):
    a = array.array("i", b)
    a.byteswap()
    return a[0]

def measure_time(f, x =None):
    if x is None:
        x = images[0][0]
    start = time.time()
    for i in range(100):
        f(x)
    return time.time() - start
