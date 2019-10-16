class Material:
    def __init__(self, bounce_sampler, bounce_resampler, color_sampler, color_resampler, physical_likelihood):
        self.bounce_sampler = bounce_sampler
        self.bounce_resampler = bounce_resampler
        self.color_sampler = color_sampler
        self.physical_likelihood = physical_likelihood

    def sample_bounce(self, **incidence_data):
        return self.bounce_sampler.sample(**incidence_data)

    def sampled_bounce_likelihood(self, bounce, **incidence_data):
        return self.bounce_sampler.likelihood(bounce, **incidence_data)

    def resample_bounce(self, old_bounce, **incidence_data):
        return self.bounce_resampler.resample(self, old_bounce, **incidence_data)

    def resampled_bounce_lr(self, new_bounce, old_bounce, **incidence_data):
        return self.bounce_resampler.lr(self, new_bounce, old_bounce, **incidence_data)

    def sample_color(self):
        return self.color_sampler.sample()

    def sampled_color_likelihood(self, color):
        return self.color_sampler.likelihood(color)

    def resample_color(self, old_color):
        return self.color_resampler.resample(old_color)

    def resampled_color_lr(self, new_color, old_color):
        return self.color_resampler.lr(self, new_color, old_color)

    def physical_likelihood(self, bounce, **incidence_data):
        return self.physical_likelihood.compute(bounce, **incidence_data)