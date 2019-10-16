from locale import Locale, Triangle, Vertex, Point
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
        return self.locale.orient(displacement)

    def unorient(self, displacement):
        return self.locale.unorient(displacement)

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
    def is_lens(self):
        return self.material.is_lens

    def sample_color(self):
        return self.material.sample_color()

    def sampled_color_likelihood(self, color):
        return self.material.sampled_color_likelihood(color)

    def sample_initial_state(self):
        point = self.sample_point()
        color = self.sample_color()
        state = point.make_state(color)
        return state

    def sampled_initial_state_likelihood(self, state):
        return self.sampled_point_likelihood(state)*self.sampled_color_likelihood(state.color)*self.emittance

    def sample_final_state(self, absorbed_state):
        point = self.sample_point
        color = absorbed_state.color
        state = point.make_state(color)
        return state


class State(Point):
    def __init__(self, location, piece, color):
        super().__init__(location, locale)
        self.color = color

    @property
    def sampled_initial_state_likelihood(self):
        self.piece.sampled_initial_state_likelihood(self)















