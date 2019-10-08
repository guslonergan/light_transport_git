import numpy as np
import math
from scipy.stats import norm
from abc import ABC, abstractmethod
import logging
from functions import normalize, extend_to_O, lens_to_hemisphere, hemisphere_to_lens
from sampler import LensSam as LensSampler
from sampler import TriangleSam as TriangleSampler
import random

TriangleSampler = TriangleSampler()



class BounceBeam:
    def __init__(self, incoming_vector, outgoing_direction, beam_color):
        self.incoming_vector = incoming_vector
        self.outgoing_direction = outgoing_direction
        self.beam_color = beam_color


class Interaction:
    def __init__(self, embeddedpoint, bouncebeam):
        self.embeddedpoint = embeddedpoint
        self.bouncebeam = bouncebeam
        self.forwards_sampling_likelihood = self.get_forwards_sampling_likelihood()
        self.backwards_sampling_likelihood = self.get_backwards_sampling_likelihood()
        self.physical_likelihood = self.get_physical_likelihood()

    def get_forwards_sampling_likelihood(self):
        return self.embeddedpoint.piece.forwards_sampling_likelihood(self.bouncebeam)

    def get_backwards_sampling_likelihood(self):
        return self.embeddedpoint.piece.backwards_sampling_likelihood(self.bouncebeam)

    def get_physical_likelihood(self):
        return self.embeddedpoint.piece.get_physical_likelihood(self.bouncebeam)


# ---------------------------------------------------------------------------


from sampler import RGB

class PhysicalLikelihoodGetter(ABC):
    @abstractmethod
    def get(self, bouncebeam):
        pass


class Lambertian(PhysicalLikelihoodGetter):
    def __init__(self, color = RGB(), emittance = 0):
        self.color = color
        self.emittance = emittance

    def get(self, bouncebeam):
        if bouncebeam.incoming_vector is 'emitted':
            a = self.emittance
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


class Atomic(PhysicalLikelihoodGetter):
    def __init__(self, color = RGB(1,1,1), emittance = 0):
        self.color = color
        self.emittance = emittance

    def get(self, bouncebeam):
        if bouncebeam.incoming_vector is 'emitted':
            a = self.emittance
        else:
            a = 1

        if bouncebeam.outgoing_direction is 'absorbed':
            b = 1
            #does a need to change in this instance?
        else:
            b = 1/(4*math.pi)

        return self.color.likelihood(bouncebeam.beam_color)*a*b


class Eyeball(PhysicalLikelihoodGetter):
    def __init__(self):
        self.emittance = 0

    def get(self, bouncebeam):
        if bouncebeam.outgoing_direction is 'absorbed':
            return 1

        else:
            raise Exception("This shouldn't happen in the current implementation... the eye has no physical extent and cannot be bounced off at the moment.")


# ---------------------------------------------------------------------------


class Medium:
    def __init__(self, is_emitter=False, is_eye=False):
        self.is_emitter = is_emitter
        self.is_eye = is_eye

#colors are only sampled or resampled at emitters
class Boundary(Medium):#TODO: make this work with full spectrum colors and refraction/attenuation indices
    def __init__(self, direction_sampler, direction_resampler, physicallikelihoodgetter, color_sampler=None):
        #color_sampler only necessary if it's an emitter... which is contained in the physicallikelihoodgetter... rearrange class structure?
        self.direction_sampler = direction_sampler
        self.direction_resampler = direction_resampler
        self.physicallikelihoodgetter = physicallikelihoodgetter
        self.emittance = self.physicallikelihoodgetter.emittance
        self.color_sampler = color_sampler

    def sample_color(self):
        return self.color_sampler.sample()

    def sampled_color_likelihood(self, color):
        return self.color_sampler.likelihood(color)

    def is_eye(self):
        return False

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


class Lens(Boundary):
    def __init__(self, x_field, y_field):# x_field, y_field are dimensions of a lens at unit distance from the eye
        self.x_field = x_field
        self.y_field = y_field
        self.direction_sampler = LensSampler(x_field, y_field)
        self.direction_resampler = LensSampler(x_field, y_field)
        self.physicallikelihoodgetter = Eyeball()

    def is_eye(self):
        return True

    def interpret(self, absorbed_bouncebeam, pixel_number):
        if absorbed_bouncebeam.outgoing_direction is 'absorbed':
            incoming_direction = normalize(absorbed_bouncebeam.incoming_vector)
            (x,y) = hemisphere_to_lens(incoming_direction)
            x_pixel = math.floor(pixel_number*(0.5 + x/self.x_field))
            y_pixel = math.floor(pixel_number*(0.5 + y/self.y_field))
            return (x_pixel, y_pixel, absorbed_bouncebeam.beam_color)


# ---------------------------------------------------------------------------


class EmbeddedPoint: # TODO consider forcing definition of physical likelihood functions on this level rather than on the level of boundaries; this would allow non-constructible optical properties
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
        self.emitters = self.get_emitters()
        self.eye = self.get_eye()

    def get_emitters(self):
        emitters = []
        for piece in self.pieces:
            if piece.boundary.physicallikelihoodgetter.emittance > 0:
                emitters += [piece]
        return emitters

    def sample_emitting_point(self):
        piece = random.choice(self.emitters)
        return piece.sample_point()

    def sampled_emitting_point_likelihood(self, embeddedpoint):
        return embeddedpoint.piece.sampled_point_likelihood(embeddedpoint)

    def get_eye(self):
        eyes = []
        for piece in self.pieces:
            if piece.boundary.is_eye():
                eyes += [piece]
        if len(eyes) == 0:
            print('There is no eye to see...')
            return None
        elif len(eyes) > 1:
            print("There are several eyes, I'll use the first I found...")
        return eyes[0]

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

    def run(self, forwards_length, backwards_length, pixel_number):
        end_embeddedpoint = self.eye.sample_point()
        start_embeddedpoint = self.sample_emitting_point()
        path = self.join(forwards_length, start_embeddedpoint, backwards_length, end_embeddedpoint)
        if path is None:
            return None
        else:
            beam_color = start_embeddedpoint.piece.boundary.sample_color()

            sampling_likelihood = end_embeddedpoint.piece.sampled_point_likelihood(end_embeddedpoint)
            sampling_likelihood *= start_embeddedpoint.piece.sampled_point_likelihood(start_embeddedpoint)
            sampling_likelihood *= start_embeddedpoint.piece.boundary.sampled_color_likelihood(beam_color)
            physical_likelihood = 1
            interaction_list = self.convert_to_interaction_list('emitted', path, 'absorbed', beam_color)
            for interaction in interaction_list:
                sampling_likelihood *= interaction.forwards_sampling_likelihood
                physical_likelihood *= interaction.physical_likelihood
            weight = physical_likelihood/sampling_likelihood
            absorbed_bouncebeam = interaction_list[-1].bouncebeam
            return (self.eye.interpret(absorbed_bouncebeam, pixel_number), weight)


class FlatPiece(Scene):
    def get_orthoframe(self):
        orthoframe = extend_to_O(self.normal)
        return orthoframe

    def hit(self, embeddedpoint, direction):
        raise Exception('Undefined.')

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

    def sample_point(self):
        raise Exception('Undefined')

    def sampled_point_likelihood(self):
        raise Exception('Undefined')

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


class Triangle(FlatPiece):
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
        self.area = self.get_area()

    def get_normal(self):
        p = self.vertices
        helper = np.cross(p[1] - p[0], p[2] - p[0])
        return normalize(helper)

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
            projection = (point-(1/np.dot(normal, direction))*np.dot(point-p[0], normal)*direction)
            for i in range(3):
                if np.dot(inwards_normals[i], projection - p[i]) < 0:
                    return None
            return EmbeddedPoint(projection, self)

    def get_area(self):
        p = self.vertices
        return np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0]))

    def sample_point(self):
        (s,t) = TriangleSampler.sample()
        p = self.vertices
        point = (1-s-t)*p[0] + s*p[1] + t*p[2]
        return EmbeddedPoint(point, self)

    def sampled_point_likelihood(self, embeddedpoint):
        if embeddedpoint.piece is self:
            return 1/self.area
        else:
            raise Exception("You shouldn't have tried to compute that likelihood... it's af a point which is not embedded in this triangle.")


class Dirac(FlatPiece):#an insubstantial point; only useful as either light source or eye
    def __init__(self, point, boundary, normal=np.array([1,0,0]), name=None, orthoframe=None):
        self.point = point
        self.boundary = boundary
        self.normal = normal
        self.name = name
        self.orthoframe = self.get_orthoframe()

    def hit(self, embeddedpoint, direction):
        return None

    def sample_point(self):
        return EmbeddedPoint(self.point, self)

    def sampled_point_likelihood(self, embeddedpoint):
        if embeddedpoint.piece is self and (embeddedpoint.point == self.point).all():
            return 1
        else:
            raise Exception('Something went wrong, getting the likelihood of a remote point...')


class DiracEye(Dirac):
    def __init__(self, point, normal, up, x_field = 5, y_field = 5, name=None, orthoframe=None):
        self.point = point
        self.normal = normalize(normal)
        self.name = name
        self.up = normalize(up)
        self.x_field = x_field
        self.y_field = y_field
        self.boundary = Lens(self.x_field, self.y_field)
        self.orthoframe = self.get_orthoframe()

    def get_orthoframe(self):
        cross = np.cross(self.normal, self.up)
        return np.array([self.up, cross, self.normal]).transpose()

    def interpret(self, absorbed_bouncebeam, pixel_number):
        return self.boundary.interpret(self.bunorient(absorbed_bouncebeam), pixel_number)



# ---------------------------------------------------------------------------



def run(scene, pixel_number, sample_number, p=0.8):
    output = dict()
    while sample_number > 0:
        forwards_length = np.random.geometric(p)
        backwards_length = np.random.geometric(p)
        do = scene.run(forwards_length, backwards_length, pixel_number)
        if do is None:
            continue
        else:
            sample_number += -1
            weight = do[1] * 1/((p**2)*((1-p)**(forwards_length+backwards_length-2)))

            if do[0][2] is 'R':
                key = (do[0][0], do[0][1], 0)
            elif do[0][2] is 'G':
                key = (do[0][0], do[0][1], 1)
            elif do[0][2] is 'B':
                key = (do[0][0], do[0][1], 2)
            else:
                raise Exception('Not an RGB pure color')


            if key in output:
                output[key] += weight
            else:
                output[key] = weight



            # if do[0][0:2] in output:
            #     if do[0][2] in output[do[0][0:2]]:
            #         output[do[0][0:2]][do[0][2]] += weight
            #     else:
            #         output[do[0][0:2]][do[0][2]] = weight
            # else:
            #     output[do[0][0:2]] = {do[0][2]:weight}




    return output







































