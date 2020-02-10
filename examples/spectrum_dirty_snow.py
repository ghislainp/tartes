import tartes
import tartes.impurities
from pylab import *


ssa = 20      # in m^2/kg
density = 300  # in kg/m^3

wavelengths = arange(300, 1000, 10)*1e-9  # from 300 to 2500nm

# pure snow
albedo_pure = tartes.albedo(wavelengths, ssa, density)

# 50ng/g of soot. Soot is the default impurity type
albedo_soot = tartes.albedo(wavelengths, ssa, density, impurities=50e-9)


# 200ng/g of Hulis.
albedo_hulis = tartes.albedo(wavelengths, ssa, density,
                             impurities=200e-9,
                             impurities_type=tartes.impurities.HULIS)


plot(wavelengths*1e9, albedo_pure, label='pure snow')
plot(wavelengths*1e9, albedo_soot, label='snow with 50 ng/g soot')
plot(wavelengths*1e9, albedo_hulis, label='snow with 200 ng/g HULIS')
legend(loc='best')
show()
