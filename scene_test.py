import unittest
import scene as item
import numpy as np
import random
import math
from scipy.stats import norm

class Scene_test(unittest.TestCase):
	def hit_test(self):
		x = np.array([100.1,0.1,0.1])
		y = np.array([0.1,100.1,0.1])
		z = np.array([0.1,0.1,100.1])
		o = np.array([0.1,0.1,0.1])
		T1 = item.Triangle( [x,y,z], None, 'T1')
		T2 = item.Triangle( [x,o,y], None, 'T2')
		T3 = item.Triangle( [y,o,z], None, 'T3')
		T4 = item.Triangle( [z,o,x], None, 'T4')
		S1 = item.Surface({T1,T2,T3,T4})
		point1 = np.array([100.1,100.1,100.1])
		point2 = np.array([0,0,0])
		point3 = np.array([1,1,1])
		embeddedpoint1 = item.EmbeddedPoint(point1, None)
		embeddedpoint2 = item.EmbeddedPoint(point2, None)
		embeddedpoint3 = item.EmbeddedPoint(point3, None)
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

		white_Lambert = item.Boundary(item.UniformHemisphere(),item.KentSphere(),item.Lambertian())

		T1 = item.Triangle( [x,y,z], white_Lambert, 'T1')
		T2 = item.Triangle( [x,o,y], white_Lambert, 'T2')
		T3 = item.Triangle( [y,o,z], white_Lambert, 'T3')
		T4 = item.Triangle( [z,o,x], white_Lambert, 'T4')

		C1 = item.Triangle( [o-50,a-50,b-50], white_Lambert, 'C1')
		C2 = item.Triangle( [a-50,a+b-50,b-50], white_Lambert, 'C2')
		C3 = item.Triangle( [o-50,c-50,a-50], white_Lambert, 'C3')
		C4 = item.Triangle( [a-50,c-50,a+c-50], white_Lambert, 'C4')
		C5 = item.Triangle( [o-50,b-50,c-50], white_Lambert, 'C5')
		C6 = item.Triangle( [c-50,b-50,b+c-50], white_Lambert, 'C6')
		C7 = item.Triangle( [a-50,a+c-50,a+b-50], white_Lambert, 'C7')
		C8 = item.Triangle( [a+b-50,a+c-50,a+b+c-50], white_Lambert, 'C8')
		C9 = item.Triangle( [b-50,a+b-50,b+c-50], white_Lambert, 'C9')
		C10 = item.Triangle( [b+c-50,a+b-50,a+b+c-50], white_Lambert, 'C10')
		C11 = item.Triangle( [c-50,b+c-50,a+c-50], white_Lambert, 'C11')
		C12 = item.Triangle( [a+c-50,b+c-50,a+b+c-50], white_Lambert, 'C12')

		S1 = item.Surface({T1,T2,T3,T4,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12})

		embeddedpoint = item.EmbeddedPoint(np.array([0,0,-50]),C1)
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

		white_Lambert = item.Boundary(item.UniformHemisphere(),item.KentSphere(),item.Lambertian())

		T1 = item.Triangle( [x,y,z], white_Lambert, 'T1')
		T2 = item.Triangle( [x,o,y], white_Lambert, 'T2')
		T3 = item.Triangle( [y,o,z], white_Lambert, 'T3')
		T4 = item.Triangle( [z,o,x], white_Lambert, 'T4')

		C1 = item.Triangle( [o-50,a-50,b-50], white_Lambert, 'C1')
		C2 = item.Triangle( [a-50,a+b-50,b-50], white_Lambert, 'C2')
		C3 = item.Triangle( [o-50,c-50,a-50], white_Lambert, 'C3')
		C4 = item.Triangle( [a-50,c-50,a+c-50], white_Lambert, 'C4')
		C5 = item.Triangle( [o-50,b-50,c-50], white_Lambert, 'C5')
		C6 = item.Triangle( [c-50,b-50,b+c-50], white_Lambert, 'C6')
		C7 = item.Triangle( [a-50,a+c-50,a+b-50], white_Lambert, 'C7')
		C8 = item.Triangle( [a+b-50,a+c-50,a+b+c-50], white_Lambert, 'C8')
		C9 = item.Triangle( [b-50,a+b-50,b+c-50], white_Lambert, 'C9')
		C10 = item.Triangle( [b+c-50,a+b-50,a+b+c-50], white_Lambert, 'C10')
		C11 = item.Triangle( [c-50,b+c-50,a+c-50], white_Lambert, 'C11')
		C12 = item.Triangle( [a+c-50,b+c-50,a+b+c-50], white_Lambert, 'C12')

		point1 = np.array([0, 0, -50])
		point2 = np.array([1, 1, -50])
		point3 = np.array([33,33,34])
		point4 = np.array([49, 49, 0])

		S1 = item.Surface({T1,T2,T3,T4,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12})

		embeddedpoint1 = item.EmbeddedPoint(point1, C1)
		embeddedpoint2 = item.EmbeddedPoint(point2, None)
		embeddedpoint3 = item.EmbeddedPoint(point3, T1)
		embeddedpoint4 = item.EmbeddedPoint(point3, None)
		embeddedpoint5 = item.EmbeddedPoint(point4, T2)

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

		x = None
		while x is None:
			x = S1.join(5,embeddedpoint1,5,embeddedpoint3)
		print(list(entry.piece.name for entry in x))
		y = S1.convert_to_bouncebeam_list('emitted', x, 'absorbed', 'B')
		print(len(x) is len(y))
		for i in range(len(y)):
			print(y[i].incoming_vector, x[i].piece.name, y[i].outgoing_direction)



x = Scene_test()
y = x.see_test()


# ---------------------------------------------------------------------------

# uniform_sphere = item.uniform_sphere()
# count = 0
# x = uniform_sphere.sample()
# n = 100000
# for i in range(0,n):
# 	y = uniform_sphere.sample()
# 	if np.dot(x,y) > 0:
# 		count = count + 1
# print('{} is approximately 0.5'.format(count/n))

# uniform_sphere = item.uniform_sphere()
# uniform_hemisphere = item.uniform_hemisphere()
# count = 0
# x = uniform_sphere.sample()
# n = 100000
# for i in range(0,n):
# 	y = uniform_hemisphere.sample()
# 	if np.dot(x,y) > 0:
# 		count = count + 1
# sample_probability = count/n
# actual_probability = (math.pi - math.acos(np.dot(x,np.array([0,0,1]))))/math.pi
# print('{} is approximately 1'.format(sample_probability/actual_probability))

# Kent_sphere = item.Kent_sphere(10000)
# print('{} is approximately [0,0,1]'.format(Kent_sphere.normalized_sample()))
# print(item.extend_to_O(np.array([0,0,1])))

# white = item.RGB(1,1,1)
# print(white.sample())
# print(white.likelihood('R'))

# ---------------------------------------------------------------------------


















