import numpy as np
import math
from abc import ABC, abstractmethod
from functions import bernoulli, monte_carlo, normalize, extend_to_O, lens_to_hemisphere, hemisphere_to_lens, normal_ppf, normal_pdf
from vectors import Displacement, Direction

#TODO: consider separating out samplers from resamplers

class Sampler(ABC):
    @abstractmethod
    def sample(self, **inputs):
        pass

    @abstractmethod
    def likelihood(self, sample, **inputs):
        pass

    def resample(self, old_sample, **inputs):
        return self.sample()

    def lr(self, new_sample, old_sample, **inputs):
        return self.likelihood(old_sample, **inputs)/self.likelihood(new_sample, **inputs)


class UniformSphere(Sampler):
    def sample(self, **incidence_data):
        def ppf_theta(t):
            return math.acos(1-2*t)
        def ppf_phi(t):
            return 2*math.pi*t
        theta = monte_carlo(ppf_theta)
        phi = monte_carlo(ppf_phi)
        return Direction(np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)]))

    def likelihood(self, direction, **incidence_data):
        return 1/(4*math.pi)

    def infinitesimal(self):
        return 'solid_angle_element'


#the hemisphere is understood to be the upper half sphere (i.e. x^2+y^2+z^2=1, z>0)
class UniformHemisphere(Sampler):
    def sample(self, **incidence_data):
        def ppf_theta(t):
            return math.acos(1-t)
        def ppf_phi(t):
            return 2*math.pi*t
        theta = monte_carlo(ppf_theta)
        phi = monte_carlo(ppf_phi)
        return Direction(np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)]))

    def likelihood(self, direction, **incidence_data):
        return 1/(2*math.pi)

    def infinitesimal(self):
        return 'solid_angle_element'


#The ellipticity parameter is set implicitly to 0 here; normal is assumed to be (0,0,1); kappa is positive, and the concentration of the distribution at the normal increases with kappa
class KentSphere(Sampler):#FIXME: this is designed to be used as a resampler; it will have big problems if used as a sampler; this is a bug
    def __init__(self, kappa=1):
        self.kappa = kappa

    def sample(self, **incidence_data):
        def ppf_phi(t):
            return 2*math.pi*t
        def ppf_u(t): #u = cos(theta)
            return 1+(1/self.kappa)*math.log(t+(1-t)*math.exp(-2*self.kappa))
        phi = monte_carlo(ppf_phi)
        u = monte_carlo(ppf_u)
        return Direction(np.array([math.sqrt(1-u**2)*math.cos(phi), math.sqrt(1-u**2)*math.sin(phi), u]))

    def likelihood(self, direction, **incidence_data):
        return math.exp(self.kappa*direction.idem.item(2))*self.kappa/(4*math.pi*math.sinh(self.kappa))

    def resample(self, old_direction, **incidence_data):
        new_direction = self.sample(**incidence_data)
        matrix = extend_to_O(old_direction.idem)
        new_direction.transform(matrix)
        return new_direction

    def lr(self, new_sample, old_sample):
        return 1

    def infinitesimal(self):
        return 'solid_angle_element'


# class LensSam(Sampler):
#     def __init__(self, x_field, y_field):
#         self.x_field = x_field
#         self.y_field = y_field

#     def sample(self):
#         def ppf_x(t):
#             return self.x_field*(t - 0.5)
#         def ppf_y(t):
#             return self.y_field*(t - 0.5)
#         x = monte_carlo(ppf_x)
#         y = monte_carlo(ppf_y)
#         return lens_to_hemisphere(x,y)

#     def likelihood(self, sample):
#         return 1

#     def infinitesimal(self):
#         return 'such that the lens at unit distance from the optical nerve is flat of area 1'


class RGB(Sampler):
    def __init__(self,R=1,G=1,B=1):
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

    def lr(self, new_sample, old_sample):
        return 1


# class FrequencySampler(Sampler):
#     def __init__(self, frequency_distribution, frequency_redistribution):
#         self.sampling_ppf = frequency_distribution.ppf
#         self.sampling_pdf = frequency_distribution.pdf
#         self.resampling_ppf = frequency_redistribution.ppf
#         self.resampling_pdf = frequency_redistribution.pdf

#     def sample(self, **inputs):
#         return monte_carlo(self.sampling_ppf)

#     def likelihood(self, frequency):
#         return self.sampling_pdf(frequency)

#     def resample(self, old_frequency):
#         return monte_carlo(self.resampling_ppf(old_frequency))

#     def lr(self, new_frequency, old_frequency):
#         return self.resampling_ppf(new_frequency)(old_frequency)/self.resampling_ppf(old_frequency)(new_frequency)


# class WhiteDistribution:
#     @staticmethod
#     def ppf(t):
#         return t*340 + 430

#     @staticmethod
#     def pdf(frequency):
#         return 1/340

#     def __init__(self):
#         self.ppf = ppf
#         self.pdf = pdf


class ScalarSampler(Sampler):
    def __init__(self, distribution, redistribution):
        self.sampling_ppf = distribution.ppf
        self.sampling_pdf = distribution.pdf
        self.resampling_ppf = redistribution.ppf
        self.resampling_pdf = redistribution.pdf

    def sample(self, **inputs):
        return monte_carlo(self.sampling_ppf)

    def likelihood(self, frequency):
        return self.sampling_pdf(frequency)

    def resample(self, old_frequency):
        return monte_carlo(self.resampling_ppf(old_frequency))

    def lr(self, new_frequency, old_frequency):
        return self.resampling_ppf(new_frequency)(old_frequency)/self.resampling_ppf(old_frequency)(new_frequency)


class WhiteDistribution:
    @property
    def ppf(self):
        def _ppf(t):
            return t*340 + 430
        return _ppf

    @property
    def pdf(self):
        def _pdf(t):
            if t < 430 or t > 770:
                return 0
            else:
                return 1/340
        return _pdf


class PureDistribution:
    def __init__(self, value):
        self.value = value

    @property
    def ppf(self):
        def _ppf(t):
            return self.value
        return _pdf

    @property
    def pdf(self):
        def _pdf(frequency):
            if frequency == self.value:
                return 1
            else:
                return 0
        return _ppf


class NormalRedistribution:
    def __init__(self, sigma):
        self.sigma = sigma

    @property
    def ppf(self):
        def _ppf(old_sample):
            def __ppf(t):
                return normal_ppf(old_sample, sigma)(t)
            return __ppf
        return _ppf

    @property
    def pdf(self):
        def _pdf(old_sample):
            def __pdf(new_sample):
                return normal_pdf(old_sample, sigma)(new_sample)
            return __pdf
        return _pdf


class WhiteSampler(ScalarSampler):
    def __init__(self):
        super().__init__(WhiteDistribution, WhiteDistribution)


class NormalResampler(ScalarSampler):
    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__(NormalRedistribution(sigma), NormalRedistribution(sigma))




