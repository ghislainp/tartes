import tartes
from pylab import *


ssa = 20      # in m^2/kg
density = 300  # in kg/m^3

wavelengths = arange(300, 2500, 10)*1e-9  # from 300 to 2500nm

albedo = tartes.albedo(wavelengths, ssa, density)

plot(wavelengths*1e9, albedo)
show()
