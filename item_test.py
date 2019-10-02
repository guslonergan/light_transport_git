import unittest
import item
import numpy as np
import random
import math
from scipy.stats import norm

class extraneous_functions_test(unittest.TestCase):
	def test(self):
		self.assertFalse(item.exists(None))
		self.assertTrue(item.exists([None,]))

class item_test(unittest.TestCase):
	def test_hit(self):
		x = np.array([100.1,0.1,0.1])
		y = np.array([0.1,100.1,0.1])
		z = np.array([0.1,0.1,100.1])
		o = np.array([0.1,0.1,0.1])
		T1 = item.triangle( [x,y,z], 'Air', 'Glass')
		T2 = item.triangle( [x,o,y], 'Air', 'Glass')
		T3 = item.triangle( [y,o,z], 'Air', 'Glass')
		T4 = item.triangle( [z,o,x], 'Air', 'Glass')
		S1 = item.composite_surface([T1,T2,T3,T4])
		point1 = np.array([100.1,100.1,100.1])
		point2 = np.array([0,0,0])
		point3 = np.array([1,1,1])
		direction = np.array([1.1,2.1,3.1])
		self.assertFalse(item.exists(S1.hit(point1,direction)))
		self.assertTrue(item.exists(S1.hit(point2,direction)))
		self.assertTrue(item.exists(S1.hit(point3,direction)))
		self.assertEqual(S1.hit_stats(point2,direction)['media']['in'],'Air')
		self.assertEqual(S1.hit_stats(point3,direction)['media']['out'],'Air')

	def test_cast(self):
		x = np.array([100.1,0.1,0.1])
		y = np.array([0.1,100.1,0.1])
		z = np.array([0.1,0.1,100.1])
		o = np.array([0.1,0.1,0.1])

		a = np.array([200,0,0])
		b = np.array([0,200,0])
		c = np.array([0,0,200])

		shift = np.array([-50,-50,-50])

		# white_Lambert = item.white_Lambert_in_air()
		white_Lambert = item.boundary(item.uniform_hemisphere(),item.Kent_sphere(1),item.Lambert)


		# air = item.air()

		# T1 = item.triangle( [x,y,z], air, white_Lambert, 'T1')
		# T2 = item.triangle( [x,o,y], air, white_Lambert, 'T2')
		# T3 = item.triangle( [y,o,z], air, white_Lambert, 'T3')
		# T4 = item.triangle( [z,o,x], air, white_Lambert, 'T4')

		# C1 = item.triangle( [o-50,a-50,b-50], air, white_Lambert, 'C1')
		# C2 = item.triangle( [a-50,a+b-50,b-50], air, white_Lambert, 'C2')
		# C3 = item.triangle( [o-50,c-50,a-50], air, white_Lambert, 'C3')
		# C4 = item.triangle( [a-50,c-50,a+c-50], air, white_Lambert, 'C4')
		# C5 = item.triangle( [o-50,b-50,c-50], air, white_Lambert, 'C5')
		# C6 = item.triangle( [c-50,b-50,b+c-50], air, white_Lambert, 'C6')
		# C7 = item.triangle( [a-50,a+c-50,a+b-50], air, white_Lambert, 'C7')
		# C8 = item.triangle( [a+b-50,a+c-50,a+b+c-50], air, white_Lambert, 'C8')
		# C9 = item.triangle( [b-50,a+b-50,b+c-50], air, white_Lambert, 'C9')
		# C10 = item.triangle( [b+c-50,a+b-50,a+b+c-50], air, white_Lambert, 'C10')
		# C11 = item.triangle( [c-50,b+c-50,a+c-50], air, white_Lambert, 'C11')
		# C12 = item.triangle( [a+c-50,b+c-50,a+b+c-50], air, white_Lambert, 'C12')

		T1 = item.triangle( [x,y,z], white_Lambert, 'T1')
		T2 = item.triangle( [x,o,y], white_Lambert, 'T2')
		T3 = item.triangle( [y,o,z], white_Lambert, 'T3')
		T4 = item.triangle( [z,o,x], white_Lambert, 'T4')

		C1 = item.triangle( [o-50,a-50,b-50], white_Lambert, 'C1')
		C2 = item.triangle( [a-50,a+b-50,b-50], white_Lambert, 'C2')
		C3 = item.triangle( [o-50,c-50,a-50], white_Lambert, 'C3')
		C4 = item.triangle( [a-50,c-50,a+c-50], white_Lambert, 'C4')
		C5 = item.triangle( [o-50,b-50,c-50], white_Lambert, 'C5')
		C6 = item.triangle( [c-50,b-50,b+c-50], white_Lambert, 'C6')
		C7 = item.triangle( [a-50,a+c-50,a+b-50], white_Lambert, 'C7')
		C8 = item.triangle( [a+b-50,a+c-50,a+b+c-50], white_Lambert, 'C8')
		C9 = item.triangle( [b-50,a+b-50,b+c-50], white_Lambert, 'C9')
		C10 = item.triangle( [b+c-50,a+b-50,a+b+c-50], white_Lambert, 'C10')
		C11 = item.triangle( [c-50,b+c-50,a+c-50], white_Lambert, 'C11')
		C12 = item.triangle( [a+c-50,b+c-50,a+b+c-50], white_Lambert, 'C12')

		S1 = item.composite_surface({T1,T2,T3,T4,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12})

		point,piece = np.array([0,0,-50]),C1
		S1.cast(20,point,piece)

x = item_test()
y = x.test_cast()


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


















