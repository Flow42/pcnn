import numpy as np
import cv2
from pcnn import pcnn
import time

# enter the path to the input image or a folder holding
# input images here
img_name = 'img'
# define the pcnn's parameters here
epochs = 20
alpha = 0.1
beta = 1.0
V = 1.0
k_size = 9
kernel = np.ones((k_size,k_size))
write_gif = True
do_rog = True
scales = (2,2)
# instantiate a pcnn using the given parameters
p = pcnn(kernel=kernel, epochs=epochs, V=V, alpha=alpha, beta=beta,
         write_gif=write_gif, do_rog=do_rog, scales=scales)

# run the pcnn on the given image(s)
# optionally the time taken is printed. If not needed comment it out.
st = time.time()
res = p.run(img_name, use_scale_space=False)
et = time.time()

print('PCNN took {} seconds'.format(et-st))
