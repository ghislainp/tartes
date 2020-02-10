import tartes
from pylab import *

# semi-infinite medium
ssa = [40, 15]       # in m^2/kg
density = [300, 350]  # in kg/m^3
thickness = [0.3, 1]  # in m

# depth at which the calculation is performed
z = arange(0, 100, 1) * 1e-2  # from 0 to 1m depth every cm

wavelengths = [350e-9, 400e-9]

for wl in wavelengths:
    actinic_flux_profile = tartes.actinic_profile(
        wl, z, ssa, density, thickness)
    semilogx(actinic_flux_profile, -z, label='%g nm' % (wl * 1e9))

xlabel('depth (m)')
ylabel('Actinic flux (W/m^2)')
legend(loc='best')
show()
