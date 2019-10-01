import unittest
import item
import numpy as np
import random
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

		a = np.array([150.1,-50.1,-50.1])
		b = np.array([-50.1,150.1,-50.1])
		c = np.array([-50.1,-50.1,150.1])

		white_Lambert = item.white_Lambert()
		air = item.air()

		T1 = item.triangle( [x,y,z], air, white_Lambert, 'T1')
		T2 = item.triangle( [x,o,y], air, white_Lambert, 'T2')
		T3 = item.triangle( [y,o,z], air, white_Lambert, 'T3')
		T4 = item.triangle( [z,o,x], air, white_Lambert, 'T4')

		C1 = item.triangle( [o,a,b], air, white_Lambert, 'C1')
		C2 = item.triangle( [a,a+b,b], air, white_Lambert, 'C2')
		C3 = item.triangle( [o,c,a], air, white_Lambert, 'C3')
		C4 = item.triangle( [a,c,a+c], air, white_Lambert, 'C4')
		C5 = item.triangle( [o,b,c], air, white_Lambert, 'C5')
		C6 = item.triangle( [c,b,b+c], air, white_Lambert, 'C6')
		C7 = item.triangle( [a,a+c,a+b], air, white_Lambert, 'C7')
		C8 = item.triangle( [a+b,a+c,a+b+c], air, white_Lambert, 'C8')
		C9 = item.triangle( [b,a+b,b+c], air, white_Lambert, 'C9')
		C10 = item.triangle( [b+c,a+b,a+b+c], air, white_Lambert, 'C10')
		C11 = item.triangle( [c,b+c,a+c], air, white_Lambert, 'C11')
		C12 = item.triangle( [a+c,b+c,a+b+c], air, white_Lambert, 'C12')

		S1 = item.composite_surface({T1,T2,T3,T4,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12})

		point,piece = np.array([50,50,0]),C1
		S1.cast(5,point,piece)

		point = np.array([105.72346923,  32.11442017, -43.73851045])
		direction = np.array([0.15017116 , 0.36324904 ,-0.91951007])
		# print(S1.hit(point,direction)['piece'].name)


		# print(C3.choose_direction())
		# print(S1.hit(point,np.array([-0.16647569,0.6414879,0.74885187]))['piece'].name)









x = item_test()
y = x.test_cast()

# white_Lambert = item.white_Lambert()

# a = white_Lambert
# b = white_Lambert
# print(a is b)


# I = item.spherical_normal_resampler()
# print(I.resample(np.array([-20,0,1]),0,0.1))























