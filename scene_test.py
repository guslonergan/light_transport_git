import unittest
import scene
import numpy as np
import random
import math
from scipy.stats import norm
import sampler
import matplotlib.pyplot as plt
import functions

class Scene_test(unittest.TestCase):
	def hit_test(self):
		x = np.array([100.1,0.1,0.1])
		y = np.array([0.1,100.1,0.1])
		z = np.array([0.1,0.1,100.1])
		o = np.array([0.1,0.1,0.1])

		white_Lambert = scene.Boundary(sampler.UniformHemisphere(),sampler.KentSphere(),scene.Lambertian(),sampler.RGB())

		T1 = scene.Triangle( [x,y,z], white_Lambert, 'T1')
		T2 = scene.Triangle( [x,o,y], white_Lambert, 'T2')
		T3 = scene.Triangle( [y,o,z], white_Lambert, 'T3')
		T4 = scene.Triangle( [z,o,x], white_Lambert, 'T4')

		S1 = scene.Surface({T1,T2,T3,T4})

		point1 = np.array([100.1,100.1,100.1])
		point2 = np.array([0,0,0])
		point3 = np.array([1,1,1])

		embeddedpoint1 = scene.EmbeddedPoint(point1, None)
		embeddedpoint2 = scene.EmbeddedPoint(point2, None)
		embeddedpoint3 = scene.EmbeddedPoint(point3, None)

		direction = np.array([1.1,2.1,3.1])

		self.assertTrue(S1.hit(embeddedpoint1, direction) is None)
		self.assertTrue(S1.hit(embeddedpoint2, direction).piece is T3)
		self.assertTrue(S1.hit(embeddedpoint3, direction).piece is T1)

	def cast_test(self):
		x = np.array([100.1,0.1,0.1])
		y = np.array([0.1,100.1,0.1])
		z = np.array([0.1,0.1,100.1])
		o = np.array([0.1,0.1,0.1])

		a = np.array([200,0,0])
		b = np.array([0,200,0])
		c = np.array([0,0,200])

		white_Lambert = scene.Boundary(sampler.UniformHemisphere(),sampler.KentSphere(),scene.Lambertian(),sampler.RGB())

		T1 = scene.Triangle( [x,y,z], white_Lambert, 'T1')
		T2 = scene.Triangle( [x,o,y], white_Lambert, 'T2')
		T3 = scene.Triangle( [y,o,z], white_Lambert, 'T3')
		T4 = scene.Triangle( [z,o,x], white_Lambert, 'T4')

		C1 = scene.Triangle( [o-50,a-50,b-50], white_Lambert, 'C1')
		C2 = scene.Triangle( [a-50,a+b-50,b-50], white_Lambert, 'C2')
		C3 = scene.Triangle( [o-50,c-50,a-50], white_Lambert, 'C3')
		C4 = scene.Triangle( [a-50,c-50,a+c-50], white_Lambert, 'C4')
		C5 = scene.Triangle( [o-50,b-50,c-50], white_Lambert, 'C5')
		C6 = scene.Triangle( [c-50,b-50,b+c-50], white_Lambert, 'C6')
		C7 = scene.Triangle( [a-50,a+c-50,a+b-50], white_Lambert, 'C7')
		C8 = scene.Triangle( [a+b-50,a+c-50,a+b+c-50], white_Lambert, 'C8')
		C9 = scene.Triangle( [b-50,a+b-50,b+c-50], white_Lambert, 'C9')
		C10 = scene.Triangle( [b+c-50,a+b-50,a+b+c-50], white_Lambert, 'C10')
		C11 = scene.Triangle( [c-50,b+c-50,a+c-50], white_Lambert, 'C11')
		C12 = scene.Triangle( [a+c-50,b+c-50,a+b+c-50], white_Lambert, 'C12')

		S1 = scene.Surface({T1,T2,T3,T4,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12})

		embeddedpoint = scene.EmbeddedPoint(np.array([0,0,-50]),C1)
		hit_list = S1.cast(20,embeddedpoint)
		self.assertTrue(len(hit_list) == 21)
		for i in range(20):
			self.assertFalse(hit_list[i].piece is hit_list[i+1].piece)
			self.assertTrue(hit_list[i].piece.boundary is hit_list[i+1].piece.boundary)

	def see_test(self):
		x = np.array([100,0,0])
		y = np.array([0,100,0])
		z = np.array([0,0,100])
		o = np.array([0,0,0])

		a = np.array([200,0,0])
		b = np.array([0,200,0])
		c = np.array([0,0,200])

		white_Lambert = scene.Boundary(sampler.UniformHemisphere(),sampler.KentSphere(),scene.Lambertian(),sampler.RGB())

		T1 = scene.Triangle( [x,y,z], white_Lambert, 'T1')
		T2 = scene.Triangle( [x,o,y], white_Lambert, 'T2')
		T3 = scene.Triangle( [y,o,z], white_Lambert, 'T3')
		T4 = scene.Triangle( [z,o,x], white_Lambert, 'T4')

		C1 = scene.Triangle( [o-50,a-50,b-50], white_Lambert, 'C1')
		C2 = scene.Triangle( [a-50,a+b-50,b-50], white_Lambert, 'C2')
		C3 = scene.Triangle( [o-50,c-50,a-50], white_Lambert, 'C3')
		C4 = scene.Triangle( [a-50,c-50,a+c-50], white_Lambert, 'C4')
		C5 = scene.Triangle( [o-50,b-50,c-50], white_Lambert, 'C5')
		C6 = scene.Triangle( [c-50,b-50,b+c-50], white_Lambert, 'C6')
		C7 = scene.Triangle( [a-50,a+c-50,a+b-50], white_Lambert, 'C7')
		C8 = scene.Triangle( [a+b-50,a+c-50,a+b+c-50], white_Lambert, 'C8')
		C9 = scene.Triangle( [b-50,a+b-50,b+c-50], white_Lambert, 'C9')
		C10 = scene.Triangle( [b+c-50,a+b-50,a+b+c-50], white_Lambert, 'C10')
		C11 = scene.Triangle( [c-50,b+c-50,a+c-50], white_Lambert, 'C11')
		C12 = scene.Triangle( [a+c-50,b+c-50,a+b+c-50], white_Lambert, 'C12')

		point1 = np.array([0, 0, -50])
		point2 = np.array([1, 1, -50])
		point3 = np.array([33,33,34])
		point4 = np.array([49, 49, 0])

		S1 = scene.Surface({T1,T2,T3,T4,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12})

		embeddedpoint1 = scene.EmbeddedPoint(point1, C1)
		embeddedpoint2 = scene.EmbeddedPoint(point2, None)
		embeddedpoint3 = scene.EmbeddedPoint(point3, T1)
		embeddedpoint4 = scene.EmbeddedPoint(point3, None)
		embeddedpoint5 = scene.EmbeddedPoint(point4, T2)

		self.assertFalse(S1.see(embeddedpoint1, embeddedpoint1))
		self.assertFalse(S1.see(embeddedpoint1, embeddedpoint2))
		self.assertFalse(S1.see(embeddedpoint2, embeddedpoint1))
		self.assertFalse(S1.see(embeddedpoint1, embeddedpoint3))
		self.assertFalse(S1.see(embeddedpoint2, embeddedpoint3))
		self.assertFalse(S1.see(embeddedpoint1, embeddedpoint4))
		self.assertFalse(S1.see(embeddedpoint2, embeddedpoint4))
		self.assertFalse(S1.see(embeddedpoint4, embeddedpoint2))
		self.assertTrue(S1.see(embeddedpoint1, embeddedpoint5))
		self.assertTrue(S1.see(embeddedpoint2, embeddedpoint5))
		self.assertTrue(S1.see(embeddedpoint5, embeddedpoint1))
		self.assertFalse(S1.see(embeddedpoint5, embeddedpoint2))

	def join_test(self):
		x = np.array([100,0,0])
		y = np.array([0,100,0])
		z = np.array([0,0,100])
		o = np.array([0,0,0])

		a = np.array([200,0,0])
		b = np.array([0,200,0])
		c = np.array([0,0,200])

		white_Lambert = scene.Boundary(sampler.UniformHemisphere(),sampler.KentSphere(),scene.Lambertian(),sampler.RGB())

		T1 = scene.Triangle( [x,y,z], white_Lambert, 'T1')
		T2 = scene.Triangle( [x,o,y], white_Lambert, 'T2')
		T3 = scene.Triangle( [y,o,z], white_Lambert, 'T3')
		T4 = scene.Triangle( [z,o,x], white_Lambert, 'T4')

		C1 = scene.Triangle( [o-50,a-50,b-50], white_Lambert, 'C1')
		C2 = scene.Triangle( [a-50,a+b-50,b-50], white_Lambert, 'C2')
		C3 = scene.Triangle( [o-50,c-50,a-50], white_Lambert, 'C3')
		C4 = scene.Triangle( [a-50,c-50,a+c-50], white_Lambert, 'C4')
		C5 = scene.Triangle( [o-50,b-50,c-50], white_Lambert, 'C5')
		C6 = scene.Triangle( [c-50,b-50,b+c-50], white_Lambert, 'C6')
		C7 = scene.Triangle( [a-50,a+c-50,a+b-50], white_Lambert, 'C7')
		C8 = scene.Triangle( [a+b-50,a+c-50,a+b+c-50], white_Lambert, 'C8')
		C9 = scene.Triangle( [b-50,a+b-50,b+c-50], white_Lambert, 'C9')
		C10 = scene.Triangle( [b+c-50,a+b-50,a+b+c-50], white_Lambert, 'C10')
		C11 = scene.Triangle( [c-50,b+c-50,a+c-50], white_Lambert, 'C11')
		C12 = scene.Triangle( [a+c-50,b+c-50,a+b+c-50], white_Lambert, 'C12')

		point1 = np.array([0, 0, -50])
		point2 = np.array([1, 1, -50])
		point3 = np.array([33,33,34])
		point4 = np.array([49, 49, 0])

		S1 = scene.Surface({T1,T2,T3,T4,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12})

		embeddedpoint1 = scene.EmbeddedPoint(point1, C1)
		embeddedpoint2 = scene.EmbeddedPoint(point2, None)
		embeddedpoint3 = scene.EmbeddedPoint(point3, T1)
		embeddedpoint4 = scene.EmbeddedPoint(point3, None)
		embeddedpoint5 = scene.EmbeddedPoint(point4, T2)

		x = None
		while x is None:
			x = S1.join(5,embeddedpoint1,5,embeddedpoint3)
		y = S1.convert_to_bouncebeam_list('emitted', x, 'absorbed', 'B')
		self.assertEqual(len(x), 12)
		self.assertEqual(len(y), 12)
		for i in range(11):
			# print(y[i].incoming_vector, x[i].piece.name, y[i].outgoing_direction, y[i].beam_color)
			self.assertTrue((y[i].outgoing_direction == scene.normalize(y[i+1].incoming_vector)).all())
		z = S1.convert_to_interaction_list('emitted', x, 'absorbed', 'B')
		# for key in z:
		# 	print(key.embeddedpoint.piece.name, key.physical_likelihood, key.forwards_sampling_likelihood, key.backwards_sampling_likelihood)
		print(S1.emitters)

	def run_test(self):
		x = np.array([100,0,0])
		y = np.array([0,100,0])
		z = np.array([0,0,100])
		o = np.array([0,0,0])

		a = np.array([200,0,0])
		b = np.array([0,200,0])
		c = np.array([0,0,200])

		white_Lambert = scene.Boundary(sampler.UniformHemisphere(),sampler.KentSphere(),scene.Lambertian(),sampler.RGB())

		white_light_atom = scene.Boundary(sampler.UniformSphere(),sampler.KentSphere(),scene.Atomic(sampler.RGB(),1),sampler.RGB())

		T1 = scene.Triangle( [x,y,z], white_Lambert, 'T1')
		T2 = scene.Triangle( [x,o,y], white_Lambert, 'T2')
		T3 = scene.Triangle( [y,o,z], white_Lambert, 'T3')
		T4 = scene.Triangle( [z,o,x], white_Lambert, 'T4')

		C1 = scene.Triangle( [o-50,a-50,b-50], white_Lambert, 'C1')
		C2 = scene.Triangle( [a-50,a+b-50,b-50], white_Lambert, 'C2')
		C3 = scene.Triangle( [o-50,c-50,a-50], white_Lambert, 'C3')
		C4 = scene.Triangle( [a-50,c-50,a+c-50], white_Lambert, 'C4')
		C5 = scene.Triangle( [o-50,b-50,c-50], white_Lambert, 'C5')
		C6 = scene.Triangle( [c-50,b-50,b+c-50], white_Lambert, 'C6')
		C7 = scene.Triangle( [a-50,a+c-50,a+b-50], white_Lambert, 'C7')
		C8 = scene.Triangle( [a+b-50,a+c-50,a+b+c-50], white_Lambert, 'C8')
		C9 = scene.Triangle( [b-50,a+b-50,b+c-50], white_Lambert, 'C9')
		C10 = scene.Triangle( [b+c-50,a+b-50,a+b+c-50], white_Lambert, 'C10')
		C11 = scene.Triangle( [c-50,b+c-50,a+c-50], white_Lambert, 'C11')
		C12 = scene.Triangle( [a+c-50,b+c-50,a+b+c-50], white_Lambert, 'C12')

		eye = scene.DiracEye(np.array([120,-20,50]), np.array([-1,1,0]), np.array([0,0,1]))
		light = scene.Dirac(np.array([120,120,50]), white_light_atom)

		S1 = scene.Surface({T1,T2,T3,T4,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,eye,light})


		pixel_number = 100

		# # #CAREFUL

		# # # x = scene.run(S1, pixel_number, 1000000)
		# # # mat = np.zeros((pixel_number, pixel_number, 3))
		# # # for key in x:
		# # # 	mat[key] = x[key]
		# # # mat = mat/mat.max()

		# # # np.save('./test_plot2', mat)
		# # # plt.imshow(mat)
		# # # plt.show()





x = Scene_test()
# x.hit_test()
# x.cast_test()
# x.see_test()
# x.join_test()

# x.run_test()

a = np.load('./test_plot2.npy')







plt.imshow(a)
plt.show()







# a = np.load('./test_plot.npy')
# index = np.unravel_index(np.argmax(a, axis=None), a.shape)
# print(index)
# print(a[12:18, 49:55, :])
# print(a[(15,52,0)])

# plt.imshow(a)
# plt.show()











