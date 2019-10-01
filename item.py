import numpy as np
import math
from scipy.stats import norm

# from random import random,randint
import random
import logging


# ---------------------------------------------------------------------------


def exists(thing):
    # thing
    try:
        if thing is None:
            return False
        else:
            return True
    except Exception:
        return True


# ---------------------------------------------------------------------------


def bernoulli():
    return random.uniform(0, 1)


def monte_carlo(ppf):
    x = bernoulli()
    return ppf(x)


def normal_ppf(m, s):
    def helper(x):
        return norm.ppf(x, m, s)
    return helper


# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------


class sampler:
    def __init__(self):
        pass

    def normalized_sample(self):
        raise Exception('Undefined.')

    def density(self, sample):
        raise Exception('Undefined.')

    def sample(self,direction = np.array([0,0,1])):
        normalized_sample = self.normalized_sample()
        return np.dot(extend_to_O(direction), normalized_sample)



class uniform_sphere(sampler):
    def sample(self,normal = 'irrelevant'):
        def ppf_theta(t):
            return math.acos(1-2*t)
        def ppf_phi(t):
            return 2*math.pi*t
        theta = monte_carlo(ppf_theta)
        phi = monte_carlo(ppf_phi)
        return np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])

    def density(self,sample):
        return 1/(4*math.pi)

    def infinitesimal(self):
        return 'solid_angle_element'


#the hemisphere is understood to be the upper half sphere (i.e. x^2+y^2+z^2=1, z>0)
class uniform_hemisphere(sampler):
    def normalized_sample(self):
        def ppf_theta(t):
            return math.acos(1-t)
        def ppf_phi(t):
            return 2*math.pi*t
        theta = monte_carlo(ppf_theta)
        phi = monte_carlo(ppf_phi)
        return np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])

    def density(self,sample):
        return 1/(2*math.pi)

    def infinitesimal(self):
        return 'solid_angle_element'


#The ellipticity parameter is set implicitly to 0 here; normal is assumed to be (0,0,1); kappa is positive, and the concentration of the distribution at the normal increases with kappa
class Kent_sphere(sampler):
    def __init__(self,kappa):
        self.kappa = kappa

    def normalized_sample(self):
        def ppf_phi(t):
            return 2*math.pi*t
        def ppf_u(t): #u = cos(theta)
            return 1+(1/self.kappa)*math.log(t+(1-t)*math.exp(-2*self.kappa))
        phi = monte_carlo(ppf_phi)
        u = monte_carlo(ppf_u)
        return np.array([math.sqrt(1-u**2)*math.cos(phi), math.sqrt(1-u**2)*math.sin(phi), u])

    def density(self,sample):
        return math.exp(self.kappa*np.array.dot(sample,np.array([0,0,1])))*self.kappa/(4*math.pi*math.sinh(self.kappa))

    def infinitesimal(self):
        return 'solid_angle_element'


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

#     def density(self,direction,sample):


#     def infinitesimal(self):
#         return 'solid_angle_element'





# ---------------------------------------------------------------------------

# class state_space:
#     def __init__(self,dimension,):
#         pass






# ---------------------------------------------------------------------------


class beam:
    def __init__(self):
        raise Exception("Undefined.")


class simple_beam(beam):
    def __init__(self, color, intensity):
        self.color = color


# ---------------------------------------------------------------------------

# the value 'attenuation' called for is exp(-k) where k is the physical attenuation coefficient
class bounce_rule:
    def __init__(self, attenuation):
        raise Exception("Undefined.")

    def density(self, args):
        raise Exception("Undefined.")


class white_Lambert(bounce_rule):
    def __init__(self, attenuation=0, sampling_sigma=1):
        self.attenuation = attenuation
        self.sampling_sigma = sampling_sigma

    def resample(self):
        pass

    def reflection_density(
        self,
        normal,
        incident_beam,
        incident_direction,
        reflected_beam,
        reflected_direction,
    ):
        # for other types of material there is also a transmission density
        if (
            incident_beam.color == reflected_beam.color
            and np.dot(incident_direction, normal) < 0
            and np.dot(reflected_direction, normal) > 0
        ):
            # return np.dot(normal,incident_beam)*(1/np.dot(normal,normal))*(1/np.linalg.norm(incident_direction))*(1/np.linalg.norm(reflected_direction))*np.dot(normal,reflected_direction)*4
            return (
                -np.dot(normal, incident_direction)
                * (1 / np.linalg.norm(normal))
                * (1 / np.linalg.norm(incident_direction))
                * (1 / math.pi)
            )
        else:
            return 0

    # takes a sample of a ray emanating from the surface; distribution is not related to the physical reflection_density above
    def reflection_sample(self, normal):
        # count = 0#FIXME
        while True:
            # print('?')
            # count = count + 1#FIXME
            output = spherical_normal_resampler().resample(
                normal, 0, self.sampling_sigma
            )
            if np.dot(output, normal) > 0:
                # print(count)#FIXME
                return output


class air(bounce_rule):
    def __init__(self, attenuation=1):
        self.attenuation = attenuation


# ---------------------------------------------------------------------------


class resampler:
    def __init__(self):
        raise Exception("Undefined.")

    def resample(self, input):
        raise Exception("Unefined")


class spherical_normal_resampler:
    def __init__(self):
        pass


    def resample(self, direction, m, s):
        ppf = normal_ppf(m, s)
        y, z = monte_carlo(ppf), monte_carlo(ppf)
        d = 4 + y ** 2 + z ** 2
        c = 4 * z / d
        b = 4 * y / d
        a = (-4 + y ** 2 + z ** 2) / d
        direction = direction / np.linalg.norm(direction)
        v1 = np.array(
            [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        )
        v2 = np.array(
            [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        )
        q, r = np.linalg.qr(np.array([-direction, v1, v1]).transpose())
        return np.dot(q, np.array([a, b, c]))


# ---------------------------------------------------------------------------


# class interaction_distribution:

# def interact(beam,angle,media):#Assume a given photon beam strikes a horizontal boundary in the x-direction at a certain angle. Samples a photon beam+direction for the interaction.
# 	if media == {'in':'Air','out':'Lambertian_White'}:
# 		pass

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

    def hit_stats(self, point, direction):
        raise Exception("Undefined.")


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

    def hit_stats(self, point, direction):
        hit = self.hit(point, direction)
        if exists(hit):
            return hit["piece"].hit_stats(point, direction)
        else:
            return None




    def cast(self, length, point, piece):
        beginning = [{'point':point, 'piece':piece}]
        # print(point,piece.name)#FIXME
        if length == 0:
            return beginning
        else:
            direction = piece.choose_direction()
            # print(direction)#FIXME
            hit = composite_surface(self-{piece}).hit(point, direction)
            if hit is None:
                return None
            else:
                remaining = self.cast(length - 1, hit['point'], hit['piece'])
                if remaining is None:
                    return None
                else:
                    return beginning + remaining

class triangle(surface):
    def __init__(self, vertices, out_medium, in_medium,name=None,stored_normal=None):
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
        self.in_medium = in_medium
        self.out_medium = out_medium
        self.name = name
        self.stored_normal = stored_normal

    def normal(self):
        if self.stored_normal is None:
            p = self.vertices
            helper = np.cross(p[1] - p[0], p[2] - p[0])
            self.stored_normal = helper/np.linalg.norm(helper)
        else:
            pass
        return self.stored_normal

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

    def media(self, direction):
        # FIXME: ...
        # TODO: ...
        # returns list of the medium the incident ray is in before it strikes the triangle followed by the medium on the other side of the triangle
        normal = self.normal()
        output = [None, None]
        index = int(np.dot(normal, direction) > 0)
        output[index] = self.out_medium
        output[1 - index] = self.in_medium
        return {"in": output[0], "out": output[1]}

    def hit_stats(self, point, direction):
        projection = self.hit(point, direction)
        if exists(projection):
            normal = self.normal()
            media = self.media(direction)
            incidence = math.acos(
                abs(np.dot(normal, direction))
                / (np.linalg.norm(normal) * np.linalg.norm(direction))
            )
            return {
                "normal": normal,
                "point": projection,
                "media": media,
                "angle": incidence,
            }
        else:
            return None

    def choose_direction(self):
        normal = self.normal()
        in_medium = self.in_medium
        in_attenuation = in_medium.attenuation
        out_medium = self.out_medium
        out_attenuation = out_medium.attenuation
        x = random.uniform(0, 1)
        if x < out_attenuation / (out_attenuation + in_attenuation):
            return in_medium.reflection_sample(normal)
        else:
            return out_medium.reflection_sample(-normal)

    # def interact(self,photon,direction):
    # 	media = self.media(direction)
    # 	normal = self.normal()
    # 	incidence = math.acos(abs(np.dot(normal,direction))/(np.linalg.norm(normal)*np.linalg.norm(direction)))


# ---------------------------------------------------------------------------


class path:
    def __init__(self):
        raise Exception("Not defined.")


# class hit_list_against_composite_surface(list, path):
#     def __init__(self, *incidences):
#         super().__init__(incidences)
#         self.points = list(incidence.get("point") for incidence in self)
#         self.pieces = list(incidence.get("piece") for incidence in self)

#     def __add__(self, other):
#         return hit_list_against_composite_surface(*(self + other))


# ---------------------------------------------------------------------------
