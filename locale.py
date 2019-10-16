import numpy as np
from vectors import Displacement, Direction
from functions import extend_to_O, bernoulli, normalize


class Locale(ABC):
    @abstractmethod
    def __init__(self, *data):
        pass

    @abstractmethod
    @property
    def orthoframe(self):
        pass

    def orient(self, displacement):
        return displacement.transform(self.orthoframe)

    def unorient(self, displacement):
        return displacement.transform(self.orthoframe.transpose())

    @abstractmethod
    def hit(self, point):
        pass

    @abstractmethod
    def sample_point(self):
        pass

    @abstractmethod
    def sampled_point_likelihood(self, point):
        pass

    def __contains__(self, point):
        return point.locale is self


class Triangle(Locale):
    def __init__(self, vertices, name=None):
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
        self.name = name

    @property
    def normal(self):
        p = self.vertices
        helper = np.cross(p[1] - p[0], p[2] - p[0])
        return normalize(helper)

    @property
    def inwards_normals(self):
        p = self.vertices
        normal = self.normal
        in_0 = np.cross(normal, p[1] - p[0])
        in_1 = np.cross(normal, p[2] - p[1])
        in_2 = np.cross(normal, p[0] - p[2])
        return [in_0, in_1, in_2]

    @property
    def orthoframe(self):
        p = self.vertices
        up = normalize(p[1] - p[0])
        cross = np.cross(self.normal, up)
        return np.array([up, cross, self.normal]).transpose()

    @property
    def area(self):
        p = self.vertices
        return np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0]))

    def hit(self, point, displacement):
        if point in self:
            return None
        else:
            p = self.vertices
            normal = self.normal
            inwards_normals = self.inwards_normals
            point = point.location
            displacement = displacement.location
            if (np.dot(normal, displacement)) * np.dot(point - p[0], normal) >= 0:
                return None
            projection = point - (1/np.dot(normal, displacement))*np.dot(point-p[0], normal)*displacement
            for i in range(3):
                if np.dot(inwards_normals[i], projection - p[i]) < 0:
                    return None
            return Point(projection, self)

    def sample_point(self):
        s = bernoulli()
        t = bernoulli()
        p = self.vertices
        point = (1-s)*p[0] + s*(1-t)*p[1] + s*t*p[2]
        return Point(point, self)

    def sampled_point_likelihood(self, point):
        if point.piece is self:
            return 1/self.area
        else:
            raise Exception("You shouldn't have tried to compute that likelihood... it's af a point which is not embedded in this triangle.")


class Vertex(Locale):#an insubstantial point
    def __init__(self, vertex, normal=np.array([0,0,1]), up=np.array([1,0,0]), name=None):
        self.vertex = vertex
        self.normal = normalize(normal)
        self.up = normalize(up)
        self.name = name

    @property
    def orthoframe(self):
        cross = np.cross(self.normal, self.up)
        return np.array([self.up, cross, self.normal]).transpose()

    def hit(self, point, displacement):
        return None

    def sample_point(self):
        return Point(self.vertex, self)

    def sampled_point_likelihood(self, point):
        if point.locale is self and (point.location == self.vertex).all():
            return 1
        else:
            raise Exception('Something went wrong, getting the likelihood of a remote point...')


class Point:
    def __init__(self, location, locale):
        self.location = location
        self.locale = locale

    @property
    def is_special(self):
        return self.location in ('created',
            'emitted',
            'absorbed',
            'annihilated')

    @property
    def sampled_point_likelihood(self):
        return self.locale.sampled_point_likelihood(self)

    def displacement(self, other):
        if self.location is 'created' and other.location is 'emitted':
            return Displacement('photon checking in...')
        elif self.location is 'emitted':
            return Displacement('photon emerging...')
        elif other.location is 'absorbed':
            return Displacement('photon merging...')
        elif self.location is 'absorbed' and other.location is 'annihilated':
            return Displacement('photon checking out...')
        else:
            return Displacement(other.location - self.location)

    def direction(self, other):
        return self.displacement(other).normalize()

    def distance(self, other):
        return self.displacement(other).length()

    # def make_state(self, color):
    #     return State(self.location, color, self.locale)










