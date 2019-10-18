from abc import ABC
import vectors
import sampler
import math

""" TODO: remember the emittance is equivalent to the physical likelihood of transitioning from
'photon checking in...' to 'photon emerging...'
Similarly lens_weight is equivalent to the physical likelihood of transitioning from 'photon checking out...' to
'photon merging...' """

class PhysicalLikelihood(ABC):
    def __init__(self, frequency_sampler, emittance=0, is_lens=False):
        self.color = frequency_sampler
        self.emittance = emittance
        self.is_lens = is_lens

    @property
    def is_emitter(self):
        return (self.emittance > 0)

    @abstractmethod
    def compute(self, outgoing_direction, frequency, incoming_displacement):
        pass


class Lambertian(PhysicalLikelihood):
    @property
    def is_lens(self):
        return False

    def compute(self, outgoing_direction, frequency, incoming_displacement):
        color_weight = self.color.likelihood(frequency)

        if incoming_displacement.idem is 'photon checking in...':
            return color_weight*self.emittance
        elif incoming_displacement.idem is 'photon merging...':
            return 0 #not a lens
        elif outgoing_direction.idem is 'photon merging...':
            return 0 #not a lens
        elif incoming_displacement.idem is 'photon emerging...':
            continue_likelihood = 1
        elif incoming_displacement.idem.item(2) < 0:
            continue_likelihood = color_weight*incoming_displacement.normalize().item(2)
        else:
            return 0

        if outgoing_direction.idem.item(2) > 0:
            this_way_likelihood = outgoing_direction.idem.item(2)
        else:
            return 0

        return continue_likelihood*this_way_likelihood/math.pi


class Atomic(PhysicalLikelihood):
    def compute(self, outgoing_direction, frequency, incoming_displacement):
        color_weight = self.color.likelihood(frequency)

        if incoming_displacement.idem is 'photon checking in...':
            return color_weight*self.emittance
        elif incoming_displacement.idem is 'photon merging...':
            return 1
        elif: outgoing_direction.idem is 'photon merging...':
            return float(self.is_lens)
        else:
            return 1/(4*math.pi)











