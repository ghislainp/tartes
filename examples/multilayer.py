import tartes
from pylab import *


ssa = [20, 15, 10]         # in m^2/kg
density = [200, 250, 300]  # in kg/m^3
thickness = [0.01, 0.10, 1000]  # thickness of each layer in meter
wavelengths = arange(300, 2500, 10)*1e-9  # from 300 to 2500 nm

albedo_3layers = tartes.albedo(wavelengths, ssa, density, thickness)

ssa = 20         # in m^2/kg
density = 300    # in kg/m^3
albedo_semiinfinite = tartes.albedo(wavelengths, ssa, density)


# alpha controls the transparency of the curves
plot(wavelengths*1e9, albedo_semiinfinite, alpha=0.7)
plot(wavelengths*1e9, albedo_3layers, alpha=0.7)
show()
