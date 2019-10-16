import numpy as np

class Displacement:
    def __init__(self, idem):
        self.idem = idem

    @property
    def is_special(self):
        return self.idem in ('photon checking in...',
            'photon emerging...',
            'photon merging...',
            'photon checking out...')

    def normalize(self):
        return Direction(self.idem)

    def length(self):
        if self.is_special:
            raise Exception("Special Displacements have no length.")
        else:
            return np.linalg.norm(self.idem)

    # def transform(self, matrix):
    #     if self.is_special:
    #         return self
    #     else:
    #         return Displacement(np.dot(matrix, self.idem))

    def transform(self, matrix):
        if self.is_special:
            return
        else:
            self.idem = np.dot(matrix, self.idem)
            return


class Direction(Displacement):
    def __init__(self, idem):
        super().__init__(idem)
        if self.is_special:
            pass
        else:
            super().__init__(self.idem / np.linalg.norm(self.idem))




