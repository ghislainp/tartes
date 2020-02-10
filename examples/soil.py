import tartes
from pylab import *

ssa = 20         # in m^2/kg
density = 250    # in kg/m^3


thickness = arange(5, 30, 5)*1e-2
wavelengths = arange(300, 1100, 10)*1e-9  # from 300 to 1100nm

for th in thickness:
    albedo = tartes.albedo(wavelengths, ssa, density, th, soilalbedo=0.2)
    plot(wavelengths*1e9, albedo, label='%g cm' % (th*100))

legend(loc='best')
xlabel('wavelength (nm)')
ylabel('albedo')
show()
