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

# x = item_test()
# y = x.test_hit()





I = item.spherical_normal_resampler()
print(I.resample(np.array([-20,0,1]),0,0.1))























