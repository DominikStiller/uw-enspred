import xesmf as xe


class Regridder:
    def __init__(self, target_grid):
        self.regridders = {}  # realm -> regridder
        self.target_grid = target_grid

    def regrid(self, realm, dataset):
        if realm not in self.regridders:
            self.regridders[realm] = xe.Regridder(
                dataset, self.target_grid, "bilinear", ignore_degenerate=True
            )
        return self.regridders[realm](dataset, keep_attrs=True)
