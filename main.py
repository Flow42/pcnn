

import numpy as np
from pcnn import pcnn
import time

if __name__ == '__main__':
	# enter the path to the input image or a folder holding
	# input images here
	import argparse
	parser = argparse.ArgumentParser(description='Run a unit-linking PCNN on an image or folder')
	parser.add_argument('input', help='Image filename or image folder')
	img_name = parser.parse_args().input
	# define the pcnn's parameters here
	epochs = 20
	alpha = 0.05
	beta = 1.0
	V = 0.99
	k_size = 7
	kernel = np.ones((k_size,k_size))
	write_gif = True
	brightness_is_normed = False
	scales = (1,1)
	# instantiate a pcnn using the given parameters
	p = pcnn(kernel=kernel, epochs=epochs, V=V, alpha=alpha, beta=beta,
	         write_gif=write_gif, brightness_is_normed=brightness_is_normed, scales=scales)

	# run the pcnn on the given image(s)
	# optionally the time taken is printed. If not needed comment it out.
	st = time.time()
	res = p.run(img_name, use_scale_space=False)
	et = time.time()


	# spatial_hist = p.get_spatial_histogram(img)
	# f, axarr = plt.subplots(1,2)
	# # f.tight_layout()

	# axarr[0].imshow(spatial_hist, 'gray')
	# axarr[1].imshow(img, 'gray')
	# plt.show()

	print('PCNN took {} seconds'.format(et-st))
