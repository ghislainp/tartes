import tartes
from pylab import *

# semi-infinite medium
ssa = [40, 15]       # in m^2/kg
density = [300, 350]  # in kg/m^3
thickness = [0.3, 1]  # in m

# depth at which the calculation is performed (in m)
z = arange(0, 100, 5)*1e-2  # from 0 to 1m depth every 5cm

wavelengths = [400e-9, 600e-9, 800e-9]  # in m

for wl in wavelengths:
    down_irr_profile, up_irr_profile = tartes.irradiance_profiles(
        wl, z, ssa, density, thickness)
    semilogx(up_irr_profile, -z, label='upwelling %g nm' % (wl*1e9))
    semilogx(down_irr_profile, -z, label='downwelling %g nm' % (wl*1e9))

xlabel('depth (m)')
ylabel('irradiance (W/m^2)')
legend(loc='best')
show()
