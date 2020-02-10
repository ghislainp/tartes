import tartes

ssa = 20      # in m^2/kg
density = 300  # in kg/m^3
# the result should be independent of the density for a semi-infinite medium

wavelength = 850e-9  # in m

albedo = tartes.albedo(wavelength, ssa, density, dir_frac=1, sza=30)

print(albedo)
