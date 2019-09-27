import numpy as np
import math
from random import random,randint
import logging


# ---------------------------------------------------------------------------

def exists(thing):
	try:
		if thing == None:
			return False
		else:
			return True
	except Exception:
		return True

class beam:
	def __init__(self):
		raise Exception('Undefined.')

class simple_beam(beam):
	def __init__(self,color,intensity):
		self.color = color
		self.intensity = intensity

# ---------------------------------------------------------------------------

class bounce_rule:#normalized so that the normal vector is vertical and the incoming beam is striking the point from above and left in the xz plane
	def __init__(self):
		raise Exception('Undefined.')
	def density(self,in_beam,in_angle,out_beam,out_direction):
		raise Exception('Undefined.')

class white_Lambert_in_air(bounce_rule):
	def __init__(self):
		pass
	def density(self,in_beam,in_angle,out_beam,out_direction):
		pass

# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------


# class interaction_distribution:

def interact(beam,angle,media):#Assume a given photon beam strikes a horizontal boundary in the x-direction at a certain angle. Samples a photon beam+direction for the interaction.
	if media == {'in':'Air','out':'Lambertian_White'}:
		pass

# ---------------------------------------------------------------------------


class item: #interface
	def __init__(self):
		raise Exception('Undefined.')
	def hit(self,point,direction):
		raise Exception('Undefined.')
	# def interact(self,photon,point,direction):
	# 	raise Exception('Undefined.')

class surface(item):
	def __init__(self):
		raise Exception('Undefined.')
	def media(self,point,direction):
		raise Exception('Undefined.')
	def hit_stats(self,point,direction):
		raise Exception('Undefined.')

class composite_surface(surface):
	def __init__(self,pieces):
		self.pieces = pieces #an iterable of pieces composing our surface
	def hit(self,point,direction):
		min_distance = math.inf
		for piece in self.pieces:
			projection = piece.hit(point,direction)
			if exists(projection):
				distance = np.linalg.norm(projection-point)
				if distance < min_distance:
					min_distance = distance
					closest_piece = piece
					closest_projection = projection
		try:
			return {'hit':closest_projection,'piece':closest_piece}
		except Exception:
			return None
		# 	try:
		# 		new_distance = np.linalg.norm(projection-point)
		# 		if new_distance < curr_distance:
		# 			curr_distance = new_distance
		# 			curr_piece = piece
		# 			curr_projection = projection
		# 	except Exception:
		# 		pass
		# return curr_projection,curr_piece

	def hit_stats(self,point,direction):
		hit = self.hit(point,direction)
		if exists(hit):
			return hit['piece'].hit_stats(point,direction)
		else:
			return None
		# try:#I don't like this
		# 	projection,piece = self.hit(point,direction)
		# 	return piece.hit_stats(point,direction)
		# except Exception:
		# 	return None

class triangle(surface):
	def __init__(self,vertices,out_medium,in_medium): #convention: outward pointing normal
		self.vertices = vertices #should be a list of three vertices
		self.in_medium = in_medium
		self.out_medium = out_medium
	def normal(self):
		p = self.vertices
		return np.cross(p[1]-p[0],p[2]-p[0])
	def inwards_normals(self):
		p = self.vertices
		normal = self.normal()
		in_0 = np.cross(normal,p[1]-p[0])
		in_1 = np.cross(normal,p[2]-p[1])
		in_2 = np.cross(normal,p[0]-p[2])
		return [in_0,in_1,in_2]
	def hit(self,point,direction):
		p = self.vertices
		normal = self.normal()
		inwards_normals = self.inwards_normals()
		if np.dot(normal,direction) == 0:
			return None
		if (1/np.dot(normal,direction))*np.dot(point-p[0],normal) >= 0:
			return None
		projection = point - (1/np.dot(normal,direction))*np.dot(point-p[0],normal)*direction
		for i in range(3):
			if np.dot(inwards_normals[i],projection-p[i])<0:
				return None
		return projection
	def media(self,direction):
	#returns list of the medium the incident ray is in before it strikes the triangle in followed by the medium on the other side of the triangle 
		normal = self.normal()
		output = [None,None]
		index = int(np.dot(normal,direction)>0)
		output[index] = self.out_medium
		output[1-index] = self.in_medium
		return {'in':output[0],'out':output[1]}
	def hit_stats(self,point,direction):
		projection = self.hit(point,direction)
		if exists(projection):
			normal = self.normal()
			media = self.media(direction)
			incidence = math.acos(abs(np.dot(normal,direction))/(np.linalg.norm(normal)*np.linalg.norm(direction)))
			return {'normal':normal,'hit':projection,'media':media,'angle':incidence}
		else:
			return None
	# def interact(self,photon,direction):
	# 	media = self.media(direction)
	# 	normal = self.normal()
	# 	incidence = math.acos(abs(np.dot(normal,direction))/(np.linalg.norm(normal)*np.linalg.norm(direction)))






# class test(item):
# 	def __init__(self,*args):
# 		super().__init__()


























