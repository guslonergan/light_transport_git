import numpy as np
import math
from scipy.stats import norm

# from random import random,randint
import random
import logging


# AUXILIARY FUNCTIONS---------------------------------------------------------------------------


def exists(thing):
    # thing
    try:
        if thing is None:
            return False
        else:
            return True
    except Exception:
        return True


# BASIC PROBABILITY STUFF---------------------------------------------------------------------------


def bernoulli():
    return random.uniform(0, 1)


def monte_carlo(ppf):
    x = bernoulli()
    return ppf(x)


def normal_ppf(m, s):
    def helper(x):
        return norm.ppf(x, m, s)
    return helper


# LINEAR ALGEBRA STUFF---------------------------------------------------------------------------


def normalize(vector):
    return vector/np.linalg.norm(vector)

def random_vector():
    return np.array([bernoulli(), bernoulli(), bernoulli()])

def extend_to_O(direction):
    direction = normalize(direction)
    M = np.array([direction, random_vector(), random_vector()]).transpose()
    q, r = np.linalg.qr(M)
    return r.item(0)*np.dot(q,np.array([[0,0,1],[0,1,0],[1,0,0]]))

def transport_to(direction, vector):
    return np.dot(extend_to_O(direction), vector)

def transport_from(direction, vector):
    return np.dot(extend_to_O(direction).transpose(), vector)


# VARIOUS DISTRIBUTIONS---------------------------------------------------------------------------

from abc import ABC, abstractmethod


class sampler(ABC):
    @abstractmethod
    def get_sample(self):
        pass

    @abstractmethod
    def likelihood(self, sample):
        pass

    def sample(self):
        got_sample = self.get_sample()
        return {'sample':got_sample, 'likelihood':self.likelihood(got_sample)}

    def get_resample(self, old_sample):
        return self.get_sample()

    def likelihood_ratio(self, new_sample, old_sample):
        return self.likelihood(old_sample)/self.likelihood(new_sample)

    def resample(self, old_sample):
        new_sample = self.get_resample(old_sample)
        return {'proposal':new_sample,'likelihood_ratio':self.likelihood_ratio(new_sample,old_sample)}

def flip(transporter, thing):
    if isinstance(thing, np.ndarray):
        return np.dot(transporter, thing)
    else:
        return thing


class uniform_sphere(sampler):
    def get_sample(self):
        def ppf_theta(t):
            return math.acos(1-2*t)
        def ppf_phi(t):
            return 2*math.pi*t
        theta = monte_carlo(ppf_theta)
        phi = monte_carlo(ppf_phi)
        return np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])

    def likelihood(self,sample):
        return 1/(4*math.pi)

    def infinitesimal(self):
        return 'solid_angle_element'


#the hemisphere is understood to be the upper half sphere (i.e. x^2+y^2+z^2=1, z>0)
class uniform_hemisphere(sampler):
    def get_sample(self):
        def ppf_theta(t):
            return math.acos(1-t)
        def ppf_phi(t):
            return 2*math.pi*t
        theta = monte_carlo(ppf_theta)
        phi = monte_carlo(ppf_phi)
        return np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])

    def likelihood(self,sample):
        return 1/(2*math.pi)

    def infinitesimal(self):
        return 'solid_angle_element'


#The ellipticity parameter is set implicitly to 0 here; normal is assumed to be (0,0,1); kappa is positive, and the concentration of the distribution at the normal increases with kappa
class Kent_sphere(sampler):#FIXME: this is designed to be used as a resampler; it will have big problems if used as a sampler; this is a bug
    def __init__(self,kappa=1):
        self.kappa = kappa

    def get_sample(self):
        def ppf_phi(t):
            return 2*math.pi*t
        def ppf_u(t): #u = cos(theta)
            return 1+(1/self.kappa)*math.log(t+(1-t)*math.exp(-2*self.kappa))
        phi = monte_carlo(ppf_phi)
        u = monte_carlo(ppf_u)
        return np.array([math.sqrt(1-u**2)*math.cos(phi), math.sqrt(1-u**2)*math.sin(phi), u])

    def likelihood(self,sample):
        sample = normalize(sample)
        return math.exp(self.kappa*np.array.dot(sample,np.array([0,0,1])))*self.kappa/(4*math.pi*math.sinh(self.kappa))

    def get_resample(self,old_sample):
        get_sample = self.get_sample()
        return transport_to(old_sample,get_sample)

    def likelihood_ratio(self, resampled, original):
        return 1

    def infinitesimal(self):
        return 'solid_angle_element'


class RGB(sampler):
    def __init__(self,R,G,B):
        total = R + G + B
        self.R = R/total
        self.G = G/total
        self.B = B/total

    def get_sample(self):
        def ppf(t):
            if t < self.R:
                return 'R'
            elif t < self.G + self.R:
                return 'G'
            else:
                return 'B'
        return monte_carlo(ppf)

    def likelihood(self,sample):
        if sample == 'R':
            return self.R
        elif sample == 'G':
            return self.G
        else:
            return self.B

    def resample(self,thing):
        return {'proposal':thing,'likelihood_ratio':1}

    def likelihood_ratio(self, resampled, original):
        return 1


#the hemisphere is understood to be the upper half sphere (i.e. x^2+y^2+z^2=1, z>0). The parameterization is by projeciton to the horizontal tangent plane; the Jacobian is z^3.
# class inverse_cube_hemisphere(sampler):
#     def __init__(self,sigma):
#         self.kappa = kappa

#     def resample(self,direction):
#         x = np.dot(direction,np.array([1,0,0]))
#         y = np.dot(direction,np.array([0,1,0]))
#         z = np.dot(direction,np.array([0,0,1]))
#         project = np.array([x/z,y/z])
#         helper = normal_ppf(0,sigma)
#         project = project + np.array([monte_carlo(helper), monte_carlo(helper)])
#         a = #too computationally intensive

#     def likelihood(self,direction,sample):


#     def infinitesimal(self):
#         return 'solid_angle_element'



# ---------------------------------------------------------------------------


class medium:
    def __init__(self, is_emitter=False, is_eye=False):
        self.is_emitter = is_emitter
        self.is_eye = is_eye

#colors are only sampled or resampled at emitters
class boundary(medium):#TODO: make this work with full spectrum colors and refraction/attenuation indices
    def __init__(self, direction_sampler, direction_resampler, physical_likelihood, emittance=0, is_eye=False):
        self.direction_sampler = direction_sampler
        self.direction_resampler = direction_resampler
        self.physical_likelihood = physical_likelihood
        self.emittance = emittance
        self.is_eye = is_eye

    def sample_direction(self):
        return self.direction_sampler.get_sample()

    def sampled_direction_likelihood(self, direction):
        return self.direction_sampler.likelihood(direction)

    def resample_direction(self, old_direction):
        return self.direction_resampler.get_resample(old_direction)

    def resampled_direction_likelihood_ratio(self, new_direction, old_direction):
        return self.direction_resampler.likelihood_ratio(new_direction, old_direction)

    def get_physical_likelihood(self, incoming_vector, outgoing_direction, beam_color):
        return self.physical_likelihood.get(incoming_vector, outgoing_direction, beam_color)

class Lambertian_likelihood:
    def __init__(self, color = RGB(1,1,1)):
        self.color = color

    def get(self, incoming_vector, outgoing_direction, beam_color):
        if incoming_vector == 'emitted':
            a = 1
        elif incoming_vector.item(2) < 0:
            a = - normalize(incoming_vector).item(2)
        else:
            a = 0

        if outgoing_direction == 'absorbed':
            b = 1
            #does a need to change in this instance?
        elif outgoing_direction.item(2) > 0:
            b = outgoing_direction.item(2)
        else:
            b = 0

        return color.likelihood(beam_color)*a*b/math.pi




# ---------------------------------------------------------------------------

class item:  # interface
    def __init__(self):
        raise Exception("Undefined.")

    def hit(self, point, direction):
        raise Exception("Undefined.")


class surface(item):
    def __init__(self):
        raise Exception("Undefined.")

    def media(self, point, direction):
        raise Exception("Undefined.")


class composite_surface(set,surface):

    def hit(self, point, direction):
        min_distance = math.inf
        for piece in self:
            projection = piece.hit(point, direction)
            if exists(projection):
                distance = np.linalg.norm(projection - point)
                if distance < min_distance:
                    min_distance = distance
                    closest_piece = piece
                    closest_projection = projection
        try:
            # print(np.linalg.norm(point-closest_projection))#FIXME
            return {"point": closest_projection, "piece": closest_piece}
        except Exception:
            return None

    def cast(self, length, point, piece):#, source, forwards = True):
        #Casts a random ray of a given length from a point of a piece, whose source is either an irradiant direction OR 'RADIANCE' OR 'EYE'
        output = [{'point':point, 'piece':piece}]
        while length > 0:
            piece = output[-1]['piece']
            point = output[-1]['point']
            sample = piece.sample_direction()
            hit = composite_surface(self - {piece}).hit(point, sample)
            if hit is None:
                return None
            else:
                output = output + [hit]
            length += -1
        return output

    def print_cast(self, length, point, piece):
        output = [{'point':point, 'piece':piece}]
        while length > 0:
            piece = output[-1]['piece']
            point = output[-1]['point']
            sample = piece.sample_direction()
            print(point,piece.name,sample)
            hit = composite_surface(self - {piece}).hit(point, sample)
            if hit is None:
                return None
            else:
                output = output + [hit]
            length += -1
        return output

    def see(self, head_point, head_piece, tail_point, tail_piece):
        direction = tail_point - head_point
        hit_piece = composite_surface(self - {head_piece}).hit(head_point, direction)['piece']
        return hit_piece == tail_piece

    def join(self, forwards_length, start_point, start_piece, backwards_length, end_point, end_piece):
        forwards_path = self.cast(forwards_length, start_point, start_piece)
        backwards_path = self.cast(backwards_length, start_point, start_piece)
        backwards_path.reverse()
        if forwards_path == None or backwards_path == None:
            return None
        else:
            head_point = forwards_path[-1]['point']
            head_piece = forwards_path[-1]['piece']
            tail_point = backwards_path[0]['point']
            tail_piece = backwards_path[0]['piece']
            if self.see(head_point, head_piece, tail_point, tail_piece):
                return forwards_path + backwards_path
            else:
                return None

    # def



class triangle(surface):
    def __init__(self, vertices, boundary, name=None, normal=None, inwards_normals=None, orthoframe=None):
        """ Docstring: short one-line description

        Followed by a longer description

        Args:
        vertices (type): what it's for
        ...

        Returns (type): ...

        We use Google-style docstrings
        """
        # convention: outward pointing normal
        self.vertices = vertices  # should be a list of three vertices
        self.boundary = boundary
        self.name = name
        self.normal = self.normal()
        self.inwards_normals = self.inwards_normals()
        self.orthoframe = self.orthoframe()

    def normal(self):
        p = self.vertices
        helper = np.cross(p[1] - p[0], p[2] - p[0])
        return normalize(helper)

    def orthoframe(self):
        orthoframe = extend_to_O(self.normal)
        return orthoframe

    def inwards_normals(self):
        p = self.vertices
        normal = self.normal
        in_0 = np.cross(normal, p[1] - p[0])
        in_1 = np.cross(normal, p[2] - p[1])
        in_2 = np.cross(normal, p[0] - p[2])
        return [in_0, in_1, in_2]

    def hit(self, point, direction):
        p = self.vertices
        normal = self.normal
        inwards_normals = self.inwards_normals
        # if np.dot(normal, direction) == 0:
        #     return None
        if (np.dot(normal, direction)) * np.dot(point - p[0], normal) >= 0:
            return None
        projection = (
            point
            - (1 / np.dot(normal, direction)) * np.dot(point - p[0], normal) * direction
        )
        for i in range(3):
            if np.dot(inwards_normals[i], projection - p[i]) < 0:
                return None
        return projection

    def orient(self, vector):
        return np.dot(self.orthoframe, vector)

    def unorient(self, vector):
        return np.dot(self.orthoframe.transpose(), vector)

    def sample_direction(self):
        return self.orient(self.boundary.sample_direction())

    def sampled_direction_likelihood(self, direction):
        return self.boundary.sampled_direction_likelihood(self.unorient(direction))

    def resample_direction(self, direction):
        return self.orient(self.boundary.resample_direction(self.unorient(direction)))

    def resampled_direction_likelihood_ratio(self, new_direction, old_direction):
        return self.boundary.resampled_direction_likelihood_ratio(self.unorient(new_direction), self.unorient(old_direction))

    def get_physical_likelihood(self, incoming_vector, outgoing_direction, beam_color):
        return self.boundary.get_physical_likelihood(self.unorient(incoming_vector), self.unorient(outgoing_direction), beam_color)


# ---------------------------------------------------------------------------


class bounce_beam:
    def __init__(self, incoming_vector, outgoing_direction, beam_color, point, piece, physical_likelihood = None, forwards_sampling_likelihood = None, backwards_sampling_likelihood = None):
        self.incoming_vector = incoming_vector
        self.outgoing_direction = outgoing_direction
        self.beam_color = beam_color
        self.point = point
        self.piece = piece
        self.physical_likelihood = self.physical_likelihood()
        # self.last_attenuation_factor = last_attenuation_factor
        self.forwards_sampling_likelihood = forwards_sampling_likelihood()
        self.backwards_sampling_likelihood = backwards_sampling_likelihood()

    def physical_likelihood(self):
        return self.piece.get_physical_likelihood(self.incoming_vector,self.outgoing_direction,self.beam_color)

    def forwards_sampling_likelihood(self):
        if outgoing_direction == 'absorbed':
            return 1#?
        else:
            return self.piece.sampled_direction_likelihood(self.outgoing_direction)

    def backwards_sampling_likelihood(self):
        if incoming_vector == 'emitted':
            return 1#?
        else:
            return self.piece.sampled_direction_likelihood(-normalize(self.incoming_vector))


# ---------------------------------------------------------------------------







