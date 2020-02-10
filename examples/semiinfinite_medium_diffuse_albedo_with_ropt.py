import tartes
from tartes import ssa

r_opt = 100e-6      # in m
density = 300       # in kg/m^3
# the result should be independent of the density for a semi-infinite medium

wavelength = 850e-9  # in m

albedo = tartes.albedo(wavelength, ssa(r_opt), density)

print(albedo)
