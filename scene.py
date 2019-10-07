import numpy as np
import math
from scipy.stats import norm
from abc import ABC, abstractmethod
import random
import logging


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

def transport_from(direction, vector):#DOWITHOUT?
    return np.dot(extend_to_O(direction).transpose(), vector)


# VARIOUS DISTRIBUTIONS---------------------------------------------------------------------------


class Sampler(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def likelihood(self, sample):
        pass

    def resample(self, old_sample):
        return self.sample()

    def likelihood_ratio(self, new_sample, old_sample):
        return self.likelihood(old_sample)/self.likelihood(new_sample)


class UniformSphere(Sampler):
    def sample(self):
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
class UniformHemisphere(Sampler):
    def sample(self):
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
class KentSphere(Sampler):#FIXME: this is designed to be used as a resampler; it will have big problems if used as a sampler; this is a bug
    def __init__(self,kappa=1):
        self.kappa = kappa

    def sample(self):
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

    def resample(self,old_sample):
        sample = self.sample()
        return transport_to(old_sample, sample)

    def likelihood_ratio(self, new_sample, old_sample):
        return 1

    def infinitesimal(self):
        return 'solid_angle_element'


class RGB(Sampler):
    def __init__(self,R,G,B):
        total = R + G + B
        self.R = R/total
        self.G = G/total
        self.B = B/total

    def sample(self):
        def ppf(t):
            if t < self.R:
                return 'R'
            elif t < self.G + self.R:
                return 'G'
            else:
                return 'B'
        return monte_carlo(ppf)

    def likelihood(self,sample):
        if sample is 'R':
            return self.R
        elif sample is 'G':
            return self.G
        else:
            return self.B

    def resample(self,thing):
        return thing

    def likelihood_ratio(self, new_sample, old_sample):
        return 1


#the hemisphere is understood to be the upper half sphere (i.e. x^2+y^2+z^2=1, z>0). The parameterization is by projeciton to the horizontal tangent plane; the Jacobian is z^3.
# class Inverse_Cube_Hemisphere(Sampler):
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


class BounceBeam:
    def __init__(self, incoming_vector, outgoing_direction, beam_color):
        self.incoming_vector = incoming_vector
        self.outgoing_direction = outgoing_direction
        self.beam_color = beam_color


class Interaction:
    def __init__(self, embeddedpoint, bouncebeam):
        self.embeddedpoint = embeddedpoint
        self.bouncebeam = bouncebeam
        self.forwards_sampling_likelihood = self.forwards_sampling_likelihood()
        self.backwards_sampling_likelihood = self.backwards_sampling_likelihood()
        self.physical_likelihood = self.physical_likelihood()

    def forwards_sampling_likelihood(self):
        return self.embeddedpoint.piece.forwards_sampling_likelihood(self.bouncebeam)

    def backwards_sampling_likelihood(self):
        return self.embeddedpoint.piece.backwards_sampling_likelihood(self.bouncebeam)

    def physical_likelihood(self):
        return self.embeddedpoint.piece.get_physical_likelihood(self.bouncebeam)


# ---------------------------------------------------------------------------


class PhysicalLikelihoodGetter(ABC):
    @abstractmethod
    def get(self, bouncebeam):
        pass

class Lambertian(PhysicalLikelihoodGetter):
    def __init__(self, color = RGB(1,1,1)):
        self.color = color

    def get(self, bouncebeam):
        if bouncebeam.incoming_vector is 'emitted':
            a = 1
        elif bouncebeam.incoming_vector.item(2) < 0:
            a = - normalize(bouncebeam.incoming_vector).item(2)
        else:
            a = 0

        if bouncebeam.outgoing_direction is 'absorbed':
            b = 1
            #does a need to change in this instance?
        elif bouncebeam.outgoing_direction.item(2) > 0:
            b = bouncebeam.outgoing_direction.item(2)
        else:
            b = 0

        return self.color.likelihood(bouncebeam.beam_color)*a*b/math.pi


# ---------------------------------------------------------------------------


class Medium:
    def __init__(self, is_emitter=False, is_eye=False):
        self.is_emitter = is_emitter
        self.is_eye = is_eye

#colors are only sampled or resampled at emitters
class Boundary(Medium):#TODO: make this work with full spectrum colors and refraction/attenuation indices
    def __init__(self, direction_sampler, direction_resampler, physicallikelihoodgetter, emittance=0, is_eye=False):
        self.direction_sampler = direction_sampler
        self.direction_resampler = direction_resampler
        self.physicallikelihoodgetter = physicallikelihoodgetter
        self.emittance = emittance
        self.is_eye = is_eye

    def sample_direction(self):
        return self.direction_sampler.sample()

    def sampled_direction_likelihood(self, direction):
        if direction is 'absorbed':
            return 1
        elif direction is 'emitted':
            return 1
        else:
            return self.direction_sampler.likelihood(direction)

    def resample_direction(self, old_direction):
        return self.direction_resampler.resample(old_direction)

    def resampled_direction_likelihood_ratio(self, new_direction, old_direction):
        return self.direction_resampler.likelihood_ratio(new_direction, old_direction)

    def forwards_sampling_likelihood(self, bouncebeam):
        return self.sampled_direction_likelihood(bouncebeam.outgoing_direction)

    def backwards_sampling_likelihood(self, bouncebeam):
        if bouncebeam.incoming_vector is 'emitted':
            argument = bouncebeam.incoming_vector
        else:
            argument = - normalize(bouncebeam.incoming_vector)
        return self.sampled_direction_likelihood(argument)

    def get_physical_likelihood(self, bouncebeam):
        return self.physicallikelihoodgetter.get(bouncebeam)


# ---------------------------------------------------------------------------


class EmbeddedPoint:
    def __init__(self, point, piece):
        self.point = point
        self.piece = piece

    def distance(self, other):
        return np.linalg.norm(self.point - other.point)


# ---------------------------------------------------------------------------


class Scene:
    def __init__(self):
        raise Exception("Undefined.")


class Surface(Scene):
    def __init__(self, pieces):
        self.pieces = pieces

    def hit(self, embeddedpoint, direction):
        min_distance = math.inf
        curr_embeddedpoint = None
        for piece in self.pieces:
            new_embeddedpoint = piece.hit(embeddedpoint, direction)
            if new_embeddedpoint is None:
                pass
            else:
                distance = embeddedpoint.distance(new_embeddedpoint)
                if distance < min_distance:
                    min_distance = distance
                    curr_embeddedpoint = new_embeddedpoint
        return curr_embeddedpoint

    def cast(self, length, embeddedpoint):
        if embeddedpoint is None:
            return None
        else:
            output = [embeddedpoint]
            while length > 0:
                length += -1
                direction = embeddedpoint.piece.sample_direction()
                embeddedpoint = self.hit(embeddedpoint, direction)
                if embeddedpoint is None:
                    return None
                else:
                    output += [embeddedpoint]
            return output

    def print_cast(self, length, embeddedpoint):
        if embeddedpoint is None:
            return None
        else:
            output = [embeddedpoint]
            while length > 0:
                length += -1
                direction = embeddedpoint.piece.sample_direction()
                print(embeddedpoint.point, embeddedpoint.piece.name, direction)
                embeddedpoint = self.hit(embeddedpoint, direction)
                if embeddedpoint is None:
                    return None
                else:
                    output += [embeddedpoint]
            print(embeddedpoint.point, embeddedpoint.piece.name)
            return output

    def see(self, head_embeddedpoint, tail_embeddedpoint):
    # Can tail_embeddedpoint be seen from head_embeddedpoint?
    # Answer is always False if tail_embeddedpoint is insubstantial (i.e. its piece is None).
    # However, the piece value of head_embeddedpoint is usually irrelevant - it only shows up in some bullshit rounding situations
        direction = tail_embeddedpoint.point - head_embeddedpoint.point
        hit_embeddedpoint = self.hit(head_embeddedpoint, direction)
        try:
            return hit_embeddedpoint.piece is tail_embeddedpoint.piece
        except AttributeError:
            return False

    def join(self, forwards_length, start_embeddedpoint, backwards_length, end_embeddedpoint):
        forwards_path = self.cast(forwards_length, start_embeddedpoint)
        backwards_path = self.cast(backwards_length, end_embeddedpoint)
        backwards_path.reverse()
        if forwards_path is None or backwards_path is None:
            return None
        else:
            head_embeddedpoint = forwards_path[-1]
            tail_embeddedpoint = backwards_path[0]
            if self.see(head_embeddedpoint, tail_embeddedpoint):
                return forwards_path + backwards_path
            else:
                return None

    def convert_to_bouncebeam_list(self, incoming_vector, intermediate_hit_list, outgoing_direction, beam_color):
        l = len(intermediate_hit_list)
        if l == 1:
            return BounceBeam(incoming_vector, outgoing_direction, beam_color)
        else:
            middle_bouncebeams = list(BounceBeam(intermediate_hit_list[i].point - intermediate_hit_list[i-1].point, normalize(intermediate_hit_list[i+1].point - intermediate_hit_list[i].point), beam_color) for i in range(1, len(intermediate_hit_list) - 1))
            return [BounceBeam(incoming_vector, normalize(intermediate_hit_list[1].point - intermediate_hit_list[0].point), beam_color)] + middle_bouncebeams + [BounceBeam(intermediate_hit_list[-1].point - intermediate_hit_list[-2].point, outgoing_direction, beam_color)]

    def convert_to_interaction_list(self, incoming_vector, intermediate_hit_list, outgoing_direction, beam_color):
        bouncebeam_list = self.convert_to_bouncebeam_list(incoming_vector, intermediate_hit_list, outgoing_direction, beam_color)
        return list(Interaction(*pair) for pair in zip(intermediate_hit_list, bouncebeam_list))





class Triangle(Scene):
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
        self.normal = self.get_normal()
        self.orthoframe = self.get_orthoframe()
        self.inwards_normals = self.get_inwards_normals()

    def get_normal(self):
        p = self.vertices
        helper = np.cross(p[1] - p[0], p[2] - p[0])
        return normalize(helper)

    def get_orthoframe(self):
        orthoframe = extend_to_O(self.normal)
        return orthoframe

    def get_inwards_normals(self):
        p = self.vertices
        normal = self.normal
        in_0 = np.cross(normal, p[1] - p[0])
        in_1 = np.cross(normal, p[2] - p[1])
        in_2 = np.cross(normal, p[0] - p[2])
        return [in_0, in_1, in_2]

    def hit(self, embeddedpoint, direction):#TODO all below
        if self is embeddedpoint.piece:
            return None
        else:
            p = self.vertices
            normal = self.normal
            inwards_normals = self.inwards_normals
            point = embeddedpoint.point
            if (np.dot(normal, direction)) * np.dot(point - p[0], normal) >= 0:
                return None
            projection = (
                point
                - (1 / np.dot(normal, direction)) * np.dot(point - p[0], normal) * direction
            )
            for i in range(3):
                if np.dot(inwards_normals[i], projection - p[i]) < 0:
                    return None
            return EmbeddedPoint(projection, self)

    def orient(self, vector):
        if vector is 'absorbed' or vector is 'emitted':
            return vector
        else:
            return np.dot(self.orthoframe, vector)

    def unorient(self, vector):
        if vector is 'absorbed' or vector is 'emitted':
            return vector
        else:
            return np.dot(self.orthoframe.transpose(), vector)

    def sample_direction(self):
        return self.orient(self.boundary.sample_direction())

    def sampled_direction_likelihood(self, direction):
        return self.boundary.sampled_direction_likelihood(self.unorient(direction))

    def resample_direction(self, direction):
        return self.orient(self.boundary.resample_direction(self.unorient(direction)))

    def resampled_direction_likelihood_ratio(self, new_direction, old_direction):
        return self.boundary.resampled_direction_likelihood_ratio(self.unorient(new_direction), self.unorient(old_direction))

    def borient(self, bouncebeam):
        return BounceBeam(self.orient(bouncebeam.incoming_vector), self.orient(bouncebeam.outgoing_direction), bouncebeam.beam_color)

    def bunorient(self, bouncebeam):
        return BounceBeam(self.unorient(bouncebeam.incoming_vector), self.unorient(bouncebeam.outgoing_direction), bouncebeam.beam_color)

    def forwards_sampling_likelihood(self, bouncebeam):
        return self.boundary.forwards_sampling_likelihood(self.bunorient(bouncebeam))

    def backwards_sampling_likelihood(self, bouncebeam):
        return self.boundary.backwards_sampling_likelihood(self.bunorient(bouncebeam))

    def get_physical_likelihood(self, bouncebeam):
        return self.boundary.physicallikelihoodgetter.get(self.bunorient(bouncebeam))


# ---------------------------------------------------------------------------










