import tensorflow as tf
import numpy as np

import PIL.Image
from io import BytesIO
from IPython.display import Image, display

def DisplayFractal(a, fmt='jpeg'):
	"""Display an array of iteration counts as a
     		colorful picture of a fractal."""
	a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
	img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
	img[a==a.max()] = 0
	a = img
	a = np.uint8(np.clip(a, 0, 255))
	f = BytesIO()t    
	print(f)
	fn = './pic/233.jpg'
	PIL.Image.fromarray(a).save(fn, fmt)
	display(Image(data=f.getvalue()))

#with tf.InteractiveSession() as sess:
sess = tf.InteractiveSession()
for i in range(1):
	Y, X = np.mgrid[ -1.3:1.3:0.005, -2:1:0.005 ]
	Z = X + 1j * Y
	xs = tf.constant( Z.astype(np.complex64) )	
	zs = tf.Variable( xs )
	ns = tf.Variable( tf.zeros_like(xs, tf.float32) )
	
	tf.global_variables_initializer().run()
	
	zs_ = zs * zs + xs
	not_diverged = tf.abs(zs_) < 4
	
	step = tf.group( zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)) )
	for i in range(200):
		step.run()
	DisplayFractal(ns.eval())
