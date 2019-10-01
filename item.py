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


def monte_carlo(ppf):
    x = random.uniform(0, 1)
    return ppf(x)


def normal_ppf(m, s):
    def helper(x):
        return norm.ppf(x, m, s)

    return helper


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
    def __init__(self, attenuation=0, sampling_sigma=0.5):
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
        while True:
            output = spherical_normal_resampler().resample(
                normal, 0, self.sampling_sigma
            )
            if np.dot(output, normal) > 0:
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
class composite_surface(surface):
    def __init__(self, pieces):
        self.pieces = pieces  # an iterable of pieces composing our surface

    def hit(self, point, direction):
        min_distance = math.inf
        for piece in self.pieces:
            projection = piece.hit(point, direction)
            if exists(projection):
                distance = np.linalg.norm(projection - point)
                if distance < min_distance:
                    min_distance = distance
                    closest_piece = piece
                    closest_projection = projection
        try:
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
        print(point,piece.name)#FIXME
        if length == 0:
            return beginning
        else:
            direction = piece.choose_direction()
            print(direction)
            hit = self.hit(point, direction)
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
            self.stored_normal = np.cross(p[1] - p[0], p[2] - p[0])
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
