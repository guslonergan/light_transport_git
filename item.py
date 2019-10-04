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

# class unit_vector

def normalize(direction):
    return direction/np.linalg.norm(direction)


def random_vector():
    return np.array([bernoulli(), bernoulli(), bernoulli()])


def extend_to_O(direction):
    direction = normalize(direction)
    #could do a 'while True try' thing here
    M = np.array([direction, random_vector(), random_vector()]).transpose()
    q, r = np.linalg.qr(M)
    return r.item(0)*np.dot(q,np.array([[0,0,1],[0,1,0],[1,0,0]]))


def transport_to(direction, vector):
    return np.dot(extend_to_O(direction), vector)

def transport_from(direction, vector):
    return np.dot(extend_to_O(direction).transpose(), vector)



# VARIOUS DISTRIBUTIONS---------------------------------------------------------------------------


class sampler:
    def __init__(self):
        pass

    def get_sample(self):
        raise Exception('Undefined.')

    def sample(self):
        get_sample = self.get_sample()
        return {'sample':get_sample, 'likelihood':self.likelihood(get_sample)}

    def likelihood(self, sample):
        raise Exception('Undefined.')

    def resample(self, thing):
        outcome = self.sample()
        return {'proposal':outcome['sample'],'likelihood_ratio':(self.likelihood(thing)/outcome['likelihood'])}

    def likelihood_ratio(self, resampled, original):
        return self.likelihood(original)/self.likelihood(resampled)


class uniform_sphere(sampler):
    def get_sample(self,normal = 'irrelevant'):
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
class Kent_sphere(sampler):
    def __init__(self,kappa):
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
        return math.exp(self.kappa*np.array.dot(sample,np.array([0,0,1])))*self.kappa/(4*math.pi*math.sinh(self.kappa))

    def resample(self,direction):
        get_sample = self.get_sample()
        return {'proposal':transport_to(direction,get_sample),'likelihood_ratio':1}

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

class boundary(medium):#TODO: make this work with full spectrum colors and refraction/attenuation indices
    def __init__(self, direction_sampler,direction_resampler, physical_likelihood_function, color=RGB(1,1,1), is_emitter=False, is_eye=False):
        self.direction_sampler = direction_sampler
        self.direction_resampler = direction_resampler
        self.physical_likelihood_function = physical_likelihood_function
        self.color = color
        self.is_emitter = is_emitter
        self.is_eye = is_eye

    def sample_direction(self, incoming_direction=None): #TODO: should this depend on the incoming direction?
        sampler = self.direction_sampler
        return sampler.sample()

    def sampled_direction_likelihood(self, sampled_direction):
        sampled_direction = normalize(sampled_direction)
        sampler = self.direction_sampler
        return sampler.likelihood(sampled_direction)

    def resample_direction(self, direction):
        resampler = self.direction_resampler
        return resampler.resample(direction)

    def resampled_direction_likelihood_ratio(self, resampled_direction, initial_direction):
        resampled_direction = normalize(resampled_direction)
        initial_direction = normalize(initial_direction)
        resampler = self.direction_resampler
        return resampler.likelihood_ratio(resampled_direction, initial_direction)

    def physical_likelihood(self, incoming_vector, outgoing_direction, beam_color):
        return physical_likelihood_function(incoming_vector, outgoing_direction)*(self.color).likelihood(beam_color)


def Lambert(incoming_vector,outgoing_direction):#we normalize the incoming vector as we don't believe in attenuation for the moment
    if incoming_direction.item(2) < 0 and outgoing_direction.item(2) > 0:
        a = normalize(incoming_direction).item(2)
        b = normalize(outgoing_direction).item(2)
        return a*b/math.pi
    else:
        return 0

# white_Lambert_in_air = boundary(uniform_hemisphere(),Kent_sphere(),Lambert)






# class white_Lambert_in_air(boundary):
#     def __init__(self,kappa=1):
#         self.kappa = kappa

#     def sample(self, incoming_direction = None):
#         outcome = uniform_hemisphere().sample()
#         likelihood = uniform_hemisphere().likelihood(outcome)
#         return {'outcome':outcome,'likelihood':likelihood}

#     def resample(self, outgoing_direction):
#         outcome = Kent_sphere(self.kappa).sample()
#         likelihood = Kent_sphere(self.kappa).likelihood(outcome)
#         return {'outcome':transport_to(outgoing_direction,outcome),'likelihood':likelihood, 'resampling_ratio':1}

#     def physical_likelihood(self, incoming_direction, outgoing_direction):
#         if incoming_direction.item(2) < 0 and outgoing_direction.item(2) > 0:
#             a = normalize(incoming_direction).item(2)
#             b = normalize(outgoing_direction).item(2)
#             return a*b/math.pi


# ---------------------------------------------------------------------------


class item:  # interface
    def __init__(self):
        raise Exception("Undefined.")

    def hit(self, point, direction):
        raise Exception("Undefined.")

    # def interact(self,photon,point,direction):
    # 	raise Exception('Undefined.')


class surface(item):
    def __init__(self):
        raise Exception("Undefined.")

    def media(self, point, direction):
        raise Exception("Undefined.")

    # def hit_stats(self, point, direction):
    #     raise Exception("Undefined.")


# may want to change this so it inherits from set,surface
class composite_surface(set,surface):
    # def __init__(self, pieces):
    #     self.pieces = pieces  # an iterable of pieces composing our surface

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

    # def hit_stats(self, point, direction):
    #     hit = self.hit(point, direction)
    #     if exists(hit):
    #         return hit["piece"].hit_stats(point, direction)
    #     else:
    #         return None

    # def cast(self, length, point, piece):
    #     beginning = [{'point':point, 'piece':piece}]
    #     likelihood = 1
    #     print(point, piece.name, likelihood)#FIXME
    #     if length == 0:
    #         return {'path':beginning, 'likelihood':1}
    #     else:
    #         outcome = piece.sample_direction()
    #         print(outcome['sample'])#FIXME
    #         hit = composite_surface(self-{piece}).hit(point, outcome['sample'])
    #         if hit is None:
    #             return None
    #         else:
    #             remaining = self.cast(length - 1, hit['point'], hit['piece'])
    #             if remaining is None:
    #                 return None
    #             else:
    #                 return {'path':beginning + remaining['path'],'likelihood':outcome['likelihood']*remaining['likelihood']}

    def cast(self, length, point, piece):#, source, forwards = True):
        #Casts a random ray of a given length from a point of a piece, whose source is either an irradiant direction OR 'RADIANCE' OR 'EYE'
        output = {'path':[{'point':point, 'piece':piece}], 'likelihood':1}
        while length > 0:
            piece = output['path'][-1]['piece']
            point = output['path'][-1]['point']
            outcome = piece.sample_direction()
            print(point,piece.name,outcome['sample'],outcome['likelihood'])
            output['likelihood'] = output['likelihood']*outcome['likelihood']
            hit = composite_surface(self-{piece}).hit(point, outcome['sample'])
            if hit is None:
                return None
            else:
                output['path'] = output['path'] + [hit]
            length += -1
        # print(output['likelihood'])
        return output

class triangle(surface):
    def __init__(self, vertices, boundary, name=None, stored_normal=None, stored_O=None):
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
        self.stored_normal = stored_normal
        self.stored_O = stored_O

    def normal(self):
        if self.stored_normal is None:
            p = self.vertices
            helper = np.cross(p[1] - p[0], p[2] - p[0])
            self.stored_normal = helper/np.linalg.norm(helper)
        else:
            pass
        return self.stored_normal

    def O(self):
        if self.stored_O is None:
            self.stored_O = extend_to_O(self.normal())
        else:
            pass
        return self.stored_O

    def inwards_normals(self):
        p = self.vertices
        normal = self.normal()
        in_0 = np.cross(normal, p[1] - p[0])
        in_1 = np.cross(normal, p[2] - p[1])
        in_2 = np.cross(normal, p[0] - p[2])
        return [in_0, in_1, in_2]

    def hit(self, point, direction):
        p = self.vertices
        normal = self.normal()
        inwards_normals = self.inwards_normals()
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

    # def hit_stats(self, point, direction):#DOWITHOUT?
    #     projection = self.hit(point, direction)
    #     if exists(projection):
    #         normal = self.normal()
    #         incidence = math.acos(
    #             abs(np.dot(normal, direction))
    #             / (np.linalg.norm(normal) * np.linalg.norm(direction))
    #         )
    #         return {
    #             "normal": normal,
    #             "point": projection,
    #             "angle": incidence,
    #         }
    #     else:
    #         return None

    def transport_to(self, vector):
        return np.dot(self.O(), vector)

    def transport_from(self, vector):
        return np.dot((self.O()).transpose(), vector)

    def sample_direction(self):#DOWITHOUT?
        # normal = self.normal()
        # in_attenuation = in_medium.attenuation
        # out_attenuation = out_medium.attenuation
        # x = random.uniform(0, 1)
        # if x < out_attenuation / (out_attenuation + in_attenuation):
        #     return in_medium.reflection_sample(normal)
        # else:
        #     return out_medium.reflection_sample(-normal)
        # O = self.O()
        boundary = self.boundary
        outcome = boundary.sample_direction()
        outcome['sample'] = self.transport_to(outcome['sample'])
        return outcome
        # return self.transport_to(boundary.sample_direction()['sample'])

    def likelihood_direction(self, direction):
        boundary = self.boundary
        direction = normalize(direction)
        direction = self.transport_from(direction)
        return boundary.sampled_direction_likelihood(direction)

    def resample_direction(self, direction):
        boundary = self.boundary
        direction = normalize(direction)
        direction = self.transport_from(direction)
        output = boundary.resample_direction(direction)
        output['proposal'] = self.transport_to(output['proposal'])
        return output

    def likelihood_ratio_direction(self, new_direction, old_direction):
        boundary = self.boundary
        new_direction = normalize(new_direction)
        old_direction = normalize(old_direction)
        new_direction = self.transport_from(new_direction)
        old_direction = self.transport_from(old_direction)
        return boundary.resampled_direction_likelihood_ratio(new_direction, old_direction)




# ---------------------------------------------------------------------------


class bounce_beam:
    def __init__(self, before_vector, after_direction, point, piece, color, stored_physical_likelihood = None, stored_last_attenuation_factor = None, stored_forwards_sampling_likelihood = None, stored_backwards_sampling_likelihood = None):
        self.before_vector = before_vector
        self.after_direction = after_direction
        self.point = point
        self.piece = piece
        self.color = color
        self.stored_physical_likelihood = stored_physical_likelihood
        self.stored_last_attenuation_factor = stored_last_attenuation_factor
        self.stored_forwards_sampling_likelihood = stored_forwards_sampling_likelihood
        self.stored_backwards_sampling_likelihood = stored_backwards_sampling_likelihood

    def physical_likelihood(self):
        if self.stored_physical_likelihood == None:
            self.stored_physical_likelihood = ((self.piece).boundary).physical_likelihood(self.before_vector,self.after_direction,self.color)
        return self.stored_physical_likelihood

    # def last_attenuation_factor(self):
    #     if self.stored_last_attenuation_factor == None:
    #         self.stored_last_attenuation_factor ==

    def forwards_sampling_likelihood(self):
        if self.stored_forwards_sampling_likelihood == None:
            self.stored_forwards_sampling_likelihood = 1 #FIXME



# class path:
    # def __init__(self, )






# ---------------------------------------------------------------------------



# JUNK



#DOWITHOUT
# class resampler:
#     def __init__(self):
#         raise Exception("Undefined.")

#     def resample(self, input):
#         raise Exception("Unefined")


# class spherical_normal_resampler:
#     def __init__(self):
#         pass


#     def resample(self, direction, m, s):
#         ppf = normal_ppf(m, s)
#         y, z = monte_carlo(ppf), monte_carlo(ppf)
#         d = 4 + y ** 2 + z ** 2
#         c = 4 * z / d
#         b = 4 * y / d
#         a = (-4 + y ** 2 + z ** 2) / d
#         direction = direction / np.linalg.norm(direction)
#         v1 = np.array(
#             [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
#         )
#         v2 = np.array(
#             [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
#         )
#         q, r = np.linalg.qr(np.array([-direction, v1, v1]).transpose())
#         return np.dot(q, np.array([a, b, c]))

# ---------------------------------------------------------------------------

#DOWITHOUT
# the value 'attenuation' called for is exp(-k) where k is the physical attenuation coefficient
# class bounce_rule:
#     def __init__(self, attenuation):
#         raise Exception("Undefined.")

#     def likelihood(self, args):
#         raise Exception("Undefined.")


# class white_Lambert(bounce_rule):
#     def __init__(self, attenuation=0, sampling_sigma=1):
#         self.attenuation = attenuation
#         self.sampling_sigma = sampling_sigma

#     def resample(self):
#         pass

#     def reflection_likelihood(
#         self,
#         normal,
#         incident_beam,
#         incident_direction,
#         reflected_beam,
#         reflected_direction,
#     ):
#         # for other types of material there is also a transmission likelihood
#         if (
#             incident_beam.color == reflected_beam.color
#             and np.dot(incident_direction, normal) < 0
#             and np.dot(reflected_direction, normal) > 0
#         ):
#             # return np.dot(normal,incident_beam)*(1/np.dot(normal,normal))*(1/np.linalg.norm(incident_direction))*(1/np.linalg.norm(reflected_direction))*np.dot(normal,reflected_direction)*4
#             return (
#                 -np.dot(normal, incident_direction)
#                 * (1 / np.linalg.norm(normal))
#                 * (1 / np.linalg.norm(incident_direction))
#                 * (1 / math.pi)
#             )
#         else:
#             return 0

#     # takes a sample of a ray emanating from the surface; distribution is not related to the physical reflection_likelihood above
#     def reflection_sample(self, normal):
#         # count = 0#FIXME
#         while True:
#             # print('?')
#             # count = count + 1#FIXME
#             output = spherical_normal_resampler().resample(
#                 normal, 0, self.sampling_sigma
#             )
#             if np.dot(output, normal) > 0:
#                 # print(count)#FIXME
#                 return output


# class air(bounce_rule):
#     def __init__(self, attenuation=1):
#         self.attenuation = attenuation


# ---------------------------------------------------------------------------



# class path:
#     def __init__(self):
#         raise Exception("Not defined.")


# class hit_list_against_composite_surface(list, path):
#     def __init__(self, *incidences):
#         super().__init__(incidences)
#         self.points = list(incidence.get("point") for incidence in self)
#         self.pieces = list(incidence.get("piece") for incidence in self)

#     def __add__(self, other):
#         return hit_list_against_composite_surface(*(self + other))


# ---------------------------------------------------------------------------
