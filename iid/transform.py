class SRGB_2_Linear(object):
    def __call__(self, sample):
        return sample ** 2.2
