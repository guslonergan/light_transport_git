import numpy as np
import math
from scipy.stats import norm
from abc import ABC, abstractmethod
import logging
from functions import normalize, extend_to_O, lens_to_hemisphere, hemisphere_to_lens, bernoulli
from sampler import LensSam as LensSampler
from displacement import Displacement, Direction
from piece import Piece, State
import random



class GeometricObject(ABC):
    @abstractmethod
    def __init__(self):
        pass


class Boundary(GeometricObject):
    def __init__(self, pieces):
        self.pieces = pieces

# Special parts of the object:

    @property
    def emitting_part(self):
        pieces = set()
        for piece in self.pieces:
            if piece.is_emitter:
                pieces.add(piece)
        return Boundary(pieces)

    @property
    def absorbing_part(self):
        pieces = set()
        for piece in self.pieces:
            if piece.is_lens:
                pieces.add(piece)
        return Boundary(pieces)

# Core geometric operations:

    def hit(self, point, displacement):
        min_distance = math.inf
        curr_point = None
        for piece in self.pieces:
            new_point = piece.hit(point, displacement)
            if new_point is None:
                pass
            else:
                distance = point.distance(new_point)
                if distance < min_distance:
                    min_distance = distance
                    curr_point = new_point
        return curr_point

    def can_see(self, head_point, tail_point):
    # Can tail_embeddedpoint be seen from head_embeddedpoint?
    # Answer is always False if tail_embeddedpoint is insubstantial (i.e. its piece is None).
    # However, the piece value of head_embeddedpoint is usually irrelevant - it only shows up in some bullshit rounding situations
        displacement = head_point.displacement(tail_point)
        distance = head_point.distance(tail_point)
        for piece in self.pieces:
            new_point = piece.hit(head_point, displacement)
            if new_point is None:
                pass
            else:
                new_distance = head_point.distance(new_point)
                if new_distance < distance:
                    if new_point.locale is tail_point.locale:
                        pass
                    else:
                        return False
                else:
                    pass
        return True

# Uniform distribution on points:

    def sample_piece(self):
        return random.choice(self.pieces)

    def sample_point(self):
        _piece = self.sample_piece()
        return _piece.sample_point()

    def sampled_point_likelihood(self, point):
        return point.sampled_point_likelihood
        # return point.piece.sampled_point_likelihood(point)

# Rules for sampling initial states:

    def sample_initial_state(self):
        _emitter = self.emitting_part.sample_piece()
        return _emitter.sample_initial_state()

    def sampled_initial_state_likelihood(self, state):
        return state.sampled_initial_state_likelihood
        # return initial_state.piece.sampled_initial_state_likelihood

# Rules for resampling initial states:

# Rules for sampling final states:

    def sample_final_state(self, absorbed_state):
        final_point = self.absorbing_part.sample_point
        final_state = final_point.make_state(absorbed_state.color)
        return final_state

    def sampled_final_state_likelihood(absorbed_state, final_state):
        return final_state.sampled_final_state_likelihood

# Rules for sampling directions:

    def sample_direction(self, incoming_displacement, state):
        return state.sample_direction(incoming_displacement)

    def sampled_direction_likelihood(self, incoming_displacement, state, outgoing_direction):
        return state.sampled_direction_likelihood(incoming_displacement, outgoing_direction)

# Rules for resampling directions:

# Rules for sampling next state:

    def sample_next_state(self, previous_state, current_state):
        if previous_state.location is 'created' and current_state.location is 'emitted':
            return self.sample_initial_state()
        elif previous_state.location is 'annihilated' and current_state.location is 'absorbed':
            return self.sample_final_state(current_state)
        else:
            incoming_displacement = displacement(previous_state, current_state)
            direction = self.sample_direction(incoming_displacement, current_state)
            next_point = self.hit(current_state, direction)
            next_state = next_point.make_state(current_state.color)
            return next_state

    def sampled_next_state_likelihood(self, previous_state, current_state, next_state):
        if previous_state.location is 'created' and current_state.location is 'emitted':
            return self.sampled_initial_state_likelihood(next_state)
        elif previous_state.location is 'annihilated' and current_state.location is 'absorbed':
            return self.sampled_final_state_likelihood(next_state)
        else:
            incoming_displacement = displacement(previous_state, current_state)
            outgoing_direction = direction(current_state, next_state)
            return self.sampled_direction_likelihood(incoming_displacement, current_state, outgoing_direction)









# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------

#color_sampler only necessary if it's an emitter...
class MetropolisBoundary:
    def __init__(self, direction_sampler, direction_resampler, physicalboundary, color_sampler=None):
        self.direction_sampler = direction_sampler
        self.direction_resampler = direction_resampler
        self.physicalboundary = physicalboundary
        self.emittance = self.physicalboundary.emittance
        self.color_sampler = color_sampler

    def sample_color(self):
        return self.color_sampler.sample()

    def sampled_color_likelihood(self, color):
        return self.color_sampler.likelihood(color)

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
        return self.physicalboundary.physical_likelihood(bouncebeam)


# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------


class Scene:
    def __init__(self):
        raise Exception("Undefined.")


class Surface(Scene):
    def __init__(self, pieces):
        self.pieces = pieces
        self.emitters = self.get_emitters()
        self.eye = self.get_eye()

    # def sample_next_embeddedpoint(self, embeddedpoint):
    #     if embeddedpoint is EmbeddedPoint('created', None):
    #         return self.sample_emitted_point
    #     elif embeddedpoint is EmbeddedPoint('annihilated', None):
    #         return self.get_eye.sample_point()
    #     else:
    #         return self.hit(embeddedpoint, embeddedpoint.piece.sample_direction())

    # def sampled_next_embeddedpoint_likelihood(self, initial_embeddedpoint, next_embeddedpoint):
    #     if initial_embeddedpoint is EmbeddedPoint('created', None):
    #         return self.sampled_emitted_point_likelihood(next_embeddedpoint)
    #     elif initial_embeddedpoint is EmbeddedPoint('annihilated', None):
    #         return

    def get_emitters(self):
        emitters = []
        for piece in self.pieces:
            if piece.metropolisboundary.physicalboundary.emittance > 0:
                emitters += [piece]
        return emitters

    def sample_emitted_point(self):
        piece = random.choice(self.emitters)
        return piece.sample_point()

    def sampled_emitted_point_likelihood(self, embeddedpoint):
        return embeddedpoint.piece.sampled_point_likelihood(embeddedpoint)

    def get_eye(self):
        eyes = []
        for piece in self.pieces:
            if piece.is_eye():
                eyes += [piece]
        if len(eyes) == 0:
            print('There is no eye to see...')
            return None
        elif len(eyes) > 1:
            print("There are several eyes, I'll use the first I found...")
        return eyes[0]


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


class Piece(Scene):
    def is_eye(self):
        return False

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
        return self.orient(self.metropolisboundary.sample_direction())

    def sampled_direction_likelihood(self, direction):
        return self.metropolisboundary.sampled_direction_likelihood(self.unorient(direction))

    def resample_direction(self, direction):
        return self.orient(self.metropolisboundary.resample_direction(self.unorient(direction)))

    def resampled_direction_likelihood_ratio(self, new_direction, old_direction):
        return self.metropolisboundary.resampled_direction_likelihood_ratio(self.unorient(new_direction), self.unorient(old_direction))

    def borient(self, bouncebeam):
        return BounceBeam(self.orient(bouncebeam.incoming_vector), self.orient(bouncebeam.outgoing_direction), bouncebeam.beam_color)

    def bunorient(self, bouncebeam):
        return BounceBeam(self.unorient(bouncebeam.incoming_vector), self.unorient(bouncebeam.outgoing_direction), bouncebeam.beam_color)

    def forwards_sampling_likelihood(self, bouncebeam):
        return self.metropolisboundary.forwards_sampling_likelihood(self.bunorient(bouncebeam))

    def backwards_sampling_likelihood(self, bouncebeam):
        return self.metropolisboundary.backwards_sampling_likelihood(self.bunorient(bouncebeam))

    def get_physical_likelihood(self, bouncebeam):
        return self.metropolisboundary.physicalboundary.physical_likelihood(self.bunorient(bouncebeam))










# ---------------------------------------------------------------------------


def run(scene, emitted_sample_number, absorbed_sample_number, p=0.8):
    output = dict()
    emitted_paths = set()
    absorbed_paths = set()
    eye = scene.get_eye()

    while emitted_sample_number > 0:
        length = np.random.geometric(p) - 1
        emitted_point = scene.sample_emitted_point()
        path = scene.cast(scene, length, emitted_point)
        if path is None:
            continue
        else:
            emitted_sample_number += -1
            beam_color = emitted_point.piece.metropolisboundary.sample_color()
            emitted_paths.add((path, beam_color))

    while absorbed_sample_number > 0:
        length = np.random.geometric(p) - 1
        absorbed_point = eye.sample_point()
        path = scene.cast(scene, length, absorbed_point)
        if path is None:
            continue
        else:
            absorbed_sample_number += -1
            absorbed_paths.add(path)

    for emitted_path in emitted_paths:
        for absorbed_path in absorbed_paths:
            if scene.see(emitted_path[0][-1], absorbed_path[0]):
                absorbed_path.reverse()
                path = emitted_path[0] + absorbed_path
                beam_color = emitted_path[1]
                interaction_list = scene.convert_to_interaction_list('emitted', path, 'absorbed', beam_color)
                physical_likelihood = 1
                for interaction in interaction_list:
                    physical_likelihood *= interaction.physical_likelihood
                sampling_likelihood = 0
                for i in range(len(interaction_list))





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

    return output







def run(scene, forwards_length, backwards_length):
    end_embeddedpoint = self.eye.sample_point()
    start_embeddedpoint = self.sample_emitting_point()
    path = self.join(forwards_length, start_embeddedpoint, backwards_length, end_embeddedpoint)
    if path is None:
        return None
    else:

        weight = physical_likelihood/sampling_likelihood
        absorbed_bouncebeam = interaction_list[-1].bouncebeam
        return (self.eye.interpret(absorbed_bouncebeam, pixel_number), weight)


















    def interpret(self, absorbed_bouncebeam, pixel_number):
        if absorbed_bouncebeam.outgoing_direction is 'absorbed':
            incoming_direction = normalize(absorbed_bouncebeam.incoming_vector)
            (x,y) = hemisphere_to_lens(incoming_direction)
            x_pixel = math.floor(pixel_number*(0.5 + x/self.x_field))
            y_pixel = math.floor(pixel_number*(0.5 + y/self.y_field))
            return (x_pixel, y_pixel, absorbed_bouncebeam.beam_color)


















