from mylocale import Locale, Triangle, Vertex, Point
from displacement import Displacement, Direction


class Piece:
    def __init__(self, locale, material):
        self.locale = locale
        self.material = material

# Quasi-inheritance from Locale

    @property
    def orthoframe(self):
        return self.locale.orthoframe

    def orient(self, displacement):
        self.locale.orient(displacement)

    def unorient(self, displacement):
        self.locale.unorient(displacement)

    def hit(self, point, displacement):
        return self.locale.hit(point)

    def sample_point(self):
        return self.locale.sample_point(point)

    def sampled_point_likelihood(self, point):
        return self.locale.sampled_point_likelihood(point)

    def __contains__(self, point):
        return (point in self.locale)

# Twisted quasi-inheritance from Material

    @property
    def is_emitter(self):
        return self.material.is_emitter

    @property
    def emittance(self):
        return self.material.emittance

    @property
    def is_lens(self):
        return self.material.is_lens

    def sample_color(self):
        return self.material.sample_color()

    def sampled_color_likelihood(self, color):
        return self.material.sampled_color_likelihood(color)

    def sample_initial_state(self):
        point = self.sample_point()
        color = self.sample_color()
        return State(point.location, self, color)

    def resample_color(self, old_color):
        return self.material.resample_color(old_color)

    def resampled_color_lr(self, new_color, old_color):
        return self.material.resampled_color_lr(new_color, old_color)

    def sampled_initial_state_likelihood(self, state):
        return self.sampled_point_likelihood(state)*self.sampled_color_likelihood(state.color)

    def sample_final_state(self, absorbed_state):
        point = self.sample_point()
        color = absorbed_state.color
        return State(point.location, self, color)

    def sampled_final_state_likelihood(self, state):
        return self.sampled_point_likelihood(state)

    def sample_bounce(self, frequency, incoming_displacement):
        self.unorient(incoming_displacement)
        bounce = self.material.sample_bounce(frequency, incoming_displacement)
        self.orient(bounce)
        return bounce

    def sampled_bounce_likelihood(self, bounce, frequency, incoming_displacement):
        self.unorient(bounce)
        self.unorient(incoming_displacement)
        return self.material.sampled_bounce_likelihood(bounce, frequency, incoming_displacement)

    def resample_bounce(self, old_bounce, frequency, incoming_displacement):
        self.unorient(old_bounce)
        self.unorient(incoming_displacement)
        new_bounce = self.material.resample_bounce(old_bounce, frequency, incoming_displacement)
        self.orient(new_bounce)
        return new_bounce

    def resampled_bounce_lr(self, new_bounce, old_bounce, frequency, incoming_displacement):
        self.unorient(new_bounce)
        self.unorient(old_bounce)
        self.unorient(incoming_displacement)
        return self.material.resampled_bounce_lr(new_bounce, old_bounce, frequency, incoming_displacement)

    def physicallikelihood(self, bounce, frequency, incoming_displacement):
        self.unorient(bounce)
        self.unorient(incoming_displacement)
        return self.material.physicallikelihood(bounce, frequency, incoming_displacement)



class State(Point):
    def __init__(self, location, piece, color):
        super().__init__(location, piece.locale)
        self.color = color
        self.piece = piece

    @property
    def sampled_initial_state_likelihood(self):
        self.piece.sampled_initial_state_likelihood(self)

    @property
    def sampled_final_state_likelihood(self):
        self.piece.sampled_final_state_likelihood(self)














