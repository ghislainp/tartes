# -*-coding:utf-8 -*
#
# TARTES, Copyright (c) 2014, Quentin Libois, Picard Ghislain
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

import numpy as np
from scipy.linalg import solve_banded


from .impurities import Soot
from .refractive_index import refice1995, refice2008, refice2016


def ssa(r_opt):
    """
return the specific surface area (SSA) from optical radius

:param r_opt: optical radius (m)
:type r_opt: scalar or array

:returns: SSA in m^2/kg
"""
    return (3 / 917.0) / r_opt


default_B0 = 1.6  # Based on Libois et al. 2014
# deduce for the b value derived from Gallet et al. 2009 SSA versus albedo measurements.
default_g0 = 0.86

# B0 and g0 are related to b factor in Kokhanosky and Zege 2004 or Picard et al. 2009 by:
# b = 4/3*sqrt(B0/(1-g0))


def albedo(wavelength, SSA, density=None, thickness=None, 
           g0=default_g0, B0=default_B0, y0=0.728, W0=0.0611,
           impurities=0.0, impurities_type=Soot, refrac_index="p2016",
           soilalbedo=0.0, dir_frac=0.0, sza=0.0):
    """
    compute the spectral albedo of a snowpack specified by the profiles of SSA and density using TARTES. The underlying interface has an albedo specified by soilalbedo (0.0 by default). For semi-infinite snowpack, use thickness = None (the default value).

    :param wavelength: wavelength (m)
    :type wavelength: array
    :param SSA: surface specific area (m^2/kg) for each layer
    :type SSA: array
    :param density: density (kg/m^3)
    :type density: array
    :param thickness: thickness of each layer (m). By default is None meaning a semi-infinite medium.
    :type thickness: array
    :param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
    :type g0: array or scalar
    :param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array like the SSA.
    :type B0: array or scalar
    :param y0: Value of y of snow grains at nr=1.3 (no unit).  See Eqs 72 and 75 in the "science" doc for the default value. y0 can be scalar (same for all layers) or an array like the SSA.
    :type y0: array or scalar
    :param W0: Value of W of snow grains at nr=1.3 (no unit). See Eqs 72 and 75 in the "science" doc for the default value. W0 can be a scalar (same for all layers) or an array like the SSA.
    :type W0: array or scalar
    :param impurities: impurities concentration (kg/kg) in each layer. It is either a constant or an array with size equal to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
    :type impurities: array or scalar
    :param impurities_type: specify the type of impurities. By defaut it is "soot". Otherwise it should be a class (or an instance) that defines the density and the imaginary part of the refractive index like the Soot class (see tartes.impurities for possible choices and to add new impurities types). It can also be a list of classes if several types of impurities are present in the snowpack. In this case, the impurities parameter must be a 2-d array.
    :type impurities_type: object or list of object
    :param soilalbedo: spectral albedo of the underlying layer (no unit). soilalbedo can be a scalar or an array like wavelength.
    :type soilalbedo: scalar or array
    :param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse incident flux)
    :type dir_frac: array
    :param sza: incident angle of direct light (degrees, 0 means nadir)
    :type sza: scalar
    :param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string: "p2016" for Picard et al. 2016, "w2008" for Warren and Brandt 2008 or "w1995" for Warren, 1984 modified as explained on Warren web site.
    :type refrac_index: string or tuple of two arrays

    :returns: a tuple with (spectral albedo, direct part of the spectral albedo (reflected part), diffuse part of the spectral albedo) for each wavelength.
"""

    mudir = np.cos(np.deg2rad(sza))

    albedo = tartes(wavelength, SSA, density, thickness, g0=g0, B0=B0, y0=y0, impurities=impurities,
                    impurities_type=impurities_type, soilalbedo=soilalbedo,
                    dir_frac=dir_frac, mudir=mudir, refrac_index=refrac_index)

    if albedo.shape == (1,):
        return float(albedo)
    else:
        return albedo


def absorption_profile(wavelength, SSA, density=None, thickness=None,
                       g0=default_g0, B0=default_B0, y0=0.728, W0=0.0611,
                       impurities=0, impurities_type=Soot,
                       soilalbedo=0.0, dir_frac=0.0, totflux=1.0, sza=0,
                       refrac_index="p2016"):
    """
compute the energy absorbed in every layer and in the soil. The parameters are the same as for the albedo function. If both the albedo and the absorption_profile is needed, a direct call to the tartes function is recommended.

:param wavelength: wavelength (m)
:type wavelength: array
:param SSA: surface specific area (m^2/kg) for each layer
:type SSA: array
:param density: density (kg/m^3)
:type density: array
:param thickness: thickness of each layer (m). By default is None meaning a semi-infinite medium.
:type thickness: array
:param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
:type g0: array or scalar
:param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array like the SSA.
:type B0: array or scalar
:param y0: Value of y of snow grains at nr=1.3 (no unit).  See Eqs 72 and 75 in the "science" doc for the default value. y0 can be scalar (same for all layers) or an array like the SSA.
:type y0: array or scalar
:param W0: Value of W of snow grains at nr=1.3 (no unit). See Eqs 72 and 75 in the "science" doc for the default value. W0 can be a scalar (same for all layers) or an array like the SSA.
:type W0: array or scalar
:param impurities: impurities concentration (kg/kg) in each layer. It is either a constant or an array with size equal to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
:type impurities: array or scalar
:param impurities_type: specify the type of impurity. By defaut it is "soot". Otherwise it should be a class (or an instance) defining the density and the imaginary part of the refractive index like the Soot class (see tartes.impurities for possible choices and to add new impurities types). It can also be a list of classes if several impurities type are present in the snowpack. In this case, the impurities parameter must be a 2-d array.
:type impurities_type: object or list of object
:param soilalbedo: spectral albedo of the underlying (no unit). albedo can be a scalar or an array like wavelength.
:type soilalbedo: scalar or array
:param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse incident flux)
:type dir_frac: array
:param totflux: total spectral incident flux (direct+diffuse) (W/m^2)
:type totflux: array
:param sza: incident angle of direct light (degrees, 0 means nadir)
:type sza: scalar
:param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string "p2016" for Picard et al. 2016 or w2008 for Warren and Brandt 2008 or w1995 for Warren, 1984 modified as noted on Warren web site.:type refrac_index: string or tuple of two arrays
:returns: spectral absorption in every layer and in the soil. The return type is an array with the first dimension for the wavelength and the second for the layers + an extra value for the soil. If the wavelength is a scalar, the first dimension is squeezed.    
    """

    mudir = np.cos(np.deg2rad(sza))

    albedo, absorption = tartes(wavelength, SSA, density, thickness,
                                g0=g0, B0=B0, y0=y0,
                                impurities=impurities, impurities_type=impurities_type,
                                soilalbedo=soilalbedo, dir_frac=dir_frac, totflux=totflux, mudir=mudir,
                                refrac_index=refrac_index,
                                compute_absorption=True)

    return np.insert(np.cumsum(thickness), 0, 0), absorption


def irradiance_profiles(wavelength, z, SSA, density=None, thickness=None,
                        g0=default_g0, B0=default_B0, y0=0.728, W0=0.0611,
                        impurities=0, impurities_type=Soot,
                        soilalbedo=0.0, dir_frac=0, totflux=1.0, sza=0,
                        refrac_index="p2016"):
    """compute the upwelling and downwelling irradiance at every depth z. The parameters are the same as for the absorption_profile function plus the depths z. If both the albedo and the absorption_profile is needed, a direct call to the tartes function is recommended.

:param wavelength: wavelength (m)
:type wavelength: array
:param z: depths at which the irradiance are calculated (m)
:type z: array
:param SSA: surface specific area (m^2/kg) for each layer
:type SSA: array
:param density: density (kg/m^3)
:type density: array
:param thickness: thickness of each layer (m). By default is None meaning a semi-infinite medium.
:type thickness: array
:param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
:type g0: array or scalar
:param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array like the SSA.
:type B0: array or scalar
:param y0: Value of y of snow grains at nr=1.3 (no unit).  See Eqs 72 and 75 in the "science" doc for the default value. y0 can be scalar (same for all layers) or an array like the SSA.
:type y0: array or scalar
:param W0: Value of W of snow grains at nr=1.3 (no unit). See Eqs 72 and 75 in the "science" doc for the default value. W0 can be a scalar (same for all layers) or an array like the SSA.
:type W0: array or scalar
:param impurities: impurities concentration (kg/kg) in each layer. It is either a constant or an array with size equal to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
:type impurities: array or scalar
:param impurities_type: specify the type of impurity. By defaut it is "soot". Otherwise it should be a class (or an instance) defining the density and the imaginary part of the refractive index like the Soot class (see tartes.impurities for possible choices and to add new impurities types). It can also be a list of classes if several impurities type are present in the snowpack. In this case, the impurities parameter must be a 2-d array.
:type impurities_type: object or list of object
:param soilalbedo: spectral albedo of the underlying (no unit). soilalbedo can be a scalar or an array like wavelength.
:type soilalbedo: scalar or array
:param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse incident flux) at every wavelength
:type dir_frac: array
:param totflux: total spectral incident flux (direct+diffuse) (W/m^2)
:type totflux: array
:param sza: solar zenith angle of direct light (degree, 0 means nadir)
:type sza: scalar
:param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string "p2016" for Picard et al. 2016 or w2008 for Warren and Brandt 2008 or w1995 for Warren, 1984 modified as noted on Warren web site.
:type refrac_index: string or tuple of two arrays

:returns: a tupple with downwelling and upwelling irradiance profiles. The return type is an array with the first dimension for the wavelength and the second for the layers. If the wavelength argument is a scalar, the first dimension is squeezed.
    """

    mudir = np.cos(np.deg2rad(sza))

    albedo, down, up = tartes(wavelength, SSA, density, thickness,
                              g0=g0, B0=B0, y0=y0,
                              impurities=impurities, impurities_type=impurities_type,
                              soilalbedo=soilalbedo, dir_frac=dir_frac, totflux=totflux, mudir=mudir,
                              refrac_index=refrac_index,
                              compute_irradiance_profiles=True, z=z)

    return down.squeeze(), up.squeeze()


def actinic_profile(wavelength, z, SSA, density, thickness=None,
                    g0=default_g0, B0=default_B0, y0=0.728, W0=0.0611,
                    impurities=0, impurities_type=Soot,
                    soilalbedo=0.0, dir_frac=0, totflux=1.0, sza=0,
                    refrac_index=None):
    """compute the actinic flux at every depth z. The parameters are the same as for the irradiance profile.

:param wavelength: wavelength (m)
:type wavelength: array
:param z: depths at which the irradiance are calculated (m)
:type z: array
:param SSA: surface specific area (m^2/kg) for each layer
:type SSA: array
:param density: density (kg/m^3)
:type density: array
:param thickness: thickness of each layer (m). By default is None meaning a semi-infinite medium.
:type thickness: array
:param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
:type g0: array or scalar
:param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array like the SSA.
:type B0: array or scalar
:param y0: Value of y of snow grains at nr=1.3 (no unit).  See Eqs 72 and 75 in the "science" doc for the default value. y0 can be scalar (same for all layers) or an array like the SSA.
:type y0: array or scalar
:param W0: Value of W of snow grains at nr=1.3 (no unit). See Eqs 72 and 75 in the "science" doc for the default value. W0 can be a scalar (same for all layers) or an array like the SSA.
:type W0: array or scalar
:param impurities: impurities concentration (kg/kg) in each layer. It is either a constant or an array with size equal to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
:type impurities: array or scalar
:param impurities_type: specify the type of impurity. By defaut it is "soot". Otherwise it should be a class (or an instance) defining the density and the imaginary part of the refractive index like the Soot class (see tartes.impurities for possible choices and to add new impurities types). It can also be a list of classes if several impurities type are present in the snowpack. In this case, the impurities parameter must be a 2-d array.
:type impurities_type: object or list of object
:param soilalbedo: spectral albedo of the underlying (no unit). soilalbedo can be a scalar or an array like wavelength.
:type soilalbedo: scalar or array
:param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse incident flux) at every wavelength
:type dir_frac: array
:param totflux: total spectral incident flux (direct+diffuse) (W/m^2)
:type totflux: array
:param sza: solar zenith angle of direct light (degree, 0 means nadir)
:type sza: scalar
:param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string "p2016" for Picard et al. 2016 or w2008 for Warren and Brandt 2008 or w1995 for Warren, 1984 modified as noted on Warren web site.
:type refrac_index: string or tuple of two arrays

:returns: the actinic flux profile. The return type is an array with the first dimension for the wavelength and the second for the layers. If the wavelength argument is a scalar, the first dimension is squeezed.
    """

    mudir = np.cos(np.deg2rad(sza))

    albedo, actinic = tartes(wavelength, SSA, density, thickness,
                             g0=g0, B0=B0, y0=y0,
                             impurities=impurities, impurities_type=impurities_type,
                             soilalbedo=soilalbedo, dir_frac=dir_frac, totflux=totflux, mudir=mudir,
                             refrac_index=refrac_index,
                             compute_actinic_profile=True, z=z)

    return actinic.squeeze()

#######################################################################################################
#
# The following functions are the core of TARTES but are as convenient to use as the previous one.
# Use it when the previous functions are insufficient or in case of performance issue. Experts only!
#
#
#######################################################################################################


def shape_parameter_variations(nr, g0, y0, W0, B0):
    """compute shape parameter variations as a function of the the refraction index with respect to the value in the visible range. These variation equations were obtained for sphere (Light Scattering Media Optics, Kokhanovsky, A., p.61) but should also apply to other shapes in a first approximation.
    see doc Section 2

    :param nr: refractive index (no unit). It is a constant array recalculated for each spectral resolution
    :type nr: array
    :param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit)
    :type g0: scalar
    :param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit)
    :type B0: scalar
    :param W0: Value of W of snow grains at nr=1.3 (no unit)
    :type W0: scalar
    :param y0: Value of y of snow grains at nr=1.3 (no unit)
    :type y0: scalar

    :returns: spectral parameters necessary to compute the asymmetry parameter and single scattering albedo of snow. For now, those parameters do not evolve with time. They depend on shape only

"""

    ginf = 0.9751 - 0.105 * (nr - 1.3)
    g00 = g0 - 0.38 * (nr - 1.3)
    B = B0 + 0.4 * (nr - 1.3)
    W = W0 + 0.17 * (nr - 1.3)
    y = y0 + 0.752 * (nr - 1.3)

    return ginf, g00, B, W, y


def impurities_co_single_scattering_albedo(wavelength, SSA, impurities_content, impurities_type):
    """return the spectral absorption of one layer due to the impurities
    see doc Section 2.6

    :param wavelength: all wavelengths of the incident light (m)
    :type wavelength: array
    :param SSA: snow specific surface area (m^2/kg)
    :param impurities: impurities is a dictionnary where keys are impurity type ("soot" or "hulis") and values are a 2-element array containing density (kg/m^3) and content (g/g)
    :type impurities: dict

    :returns: co single scattering albedo of impurities
"""

    if impurities_content is None:
        return 0

    def one_species_co_albedo(wavelength, SSA, impurities_content, impurities_type):
        """return the co-albedo for on species"""
        if impurities_content <= 0:
            return 0.0

        elif hasattr(impurities_type, "refractive_index_imag"):
            assert not hasattr(impurities_type, "absorption_coefficient")
            abs_impurities = -impurities_type.refractive_index_imag(wavelength)
            density_impurities = impurities_type.density

            # Eq (9)
            return 12 * np.pi / (wavelength * SSA) * impurities_content / density_impurities * abs_impurities

        elif hasattr(impurities_type, "MAE"):  # added by Ghislain
            mae_impurities = impurities_type.MAE(wavelength)  # in m^-1 / (kg m^-3)

            # density could be remove because it is
            # return 2/(density*SSA) * mae_impurities * impurities_content * density  # Eq (73) and (inline between 77 and 78)
            # Eq (73) and (inline between 77 and 78) (density is cancelled)
            return 2.0 / SSA * mae_impurities * impurities_content
        else:
            raise Exception("the impurities class is not well defined.")

    if hasattr(impurities_type, '__iter__'):
        cossalb = 0.0
        for i, species in enumerate(impurities_type):
            cossalb += one_species_co_albedo(wavelength,
                                             SSA, impurities_content[i], species)
    else:
        cossalb = one_species_co_albedo(
            wavelength, SSA, impurities_content, impurities_type)

    return cossalb


def single_scattering_optical_parameters(wavelength, refrac_index, SSA, rho,
                                         impurities, impurities_type, g0, y0, W0, B0):
    """return single scattering parameters of one layer
    see doc Section 2.3, 2.5, 2.6

    :param wavelength: wavelength (m)
    :type wavelength: array
    :param refrac_index: real and imag part of the refractive index (no unit)
    :type refrac_index: array
    :param SSA: snow specific surface area (m^2/kg) of one layer
    :type SSA: scalar
    :param rho: snow density (kg/m^3) of one layer
    :type rho: scalar
    :param impurities: impurities is a dictionnary where keys are impurity type ("soot" or "hulis") and values are a 2-element array containing density (kg/m^3) and content (g/g)
    :type impurities: dict

    :returns: total single scattering albedo and asymmetry factor
"""

    n, abs_ice = refrac_index            # determination of ice refractive index
    c = 24 * np.pi * abs_ice / (917. * wavelength) / \
        SSA    # 917 is ice density

    # calculation of the spectral asymmetry parameter of snow
    ginf, g00, B, W, y = shape_parameter_variations(n, g0, y0, W0, B0)
    g = ginf - (ginf - g00) * np.exp(- y * c)

    # co- single scattering albedo of pure snow
    phi = 2. / 3 * B / (1 - W)
    cossalb = 0.5 * (1 - W) * (1 - np.exp(- c * phi))

    # adding co- single scattering albedo for impureties
    cossalb += impurities_co_single_scattering_albedo(
        wavelength, SSA, impurities, impurities_type)

    ssalb = 1.0 - cossalb

    return ssalb, g


def infinite_medium_optical_parameters(ssalb, g):
    """return albedo and kestar using Delta-Eddington Approximation (The Delta-Eddington Approximation of Radiative Flux Transfer, Jospeh et al (1976)).
    The fluxes in the snowpack depend on these 2 quantities
    see doc section 1.4

    :param ssalb: single scattering albedo (no unit)
    :type ssalb: array
    :param g: asymmetry factor (no unit)
    :type g: array

    :returns: albedo and normalised AFEC
 """

    g_star = g / (1 + g)
    ssalb_star = ssalb * (1 - g**2) / (1 - g**2 * ssalb)

    # Jimenez-Aquino, J. and Varela, J. R., (2005)
    gamma1 = 0.25 * (7 - ssalb_star * (4 + 3 * g_star))
    gamma2 = -0.25 * (1 - ssalb_star * (4 - 3 * g_star))
    kestar = np.sqrt(gamma1**2 - gamma2**2)
    albedo = (gamma1 - kestar) / gamma2

    return albedo, kestar


def taustar_vector(SSA, density, thickness, ssalb, g, kestar):
    """compute the taustar and dtaustar of the snowpack, the optical depth of each layer and cumulated optical depth
    see doc Section 1.2, 1.8, 2.4

    :param SSA: snow specific surface area (m^2/kg) of one layer
    :type SSA: array
    :param density: vertical profile of density (kg/m^3)
    :type density: array
    :param thickness: thickness of each layers (m)
    :type thickness: array
    :param ssalb: single scattering albedo (no unit)
    :type ssalb: array
    :param g: asymmetry factor (no unit)
    :type g: array
    :param kestar: delta Eddington asymptotic flux extinction coefficient (no unit)
    :type kestar: array

    :returns: optical depth of each layer (unbounded + bounded) and cumulated optical depth (no unit)
"""
    sigext = density * SSA / 2
    # delta-Eddington variable change
    dtaustar_ub = sigext * thickness[np.newaxis, :] * (1 - ssalb * g**2)

    maximum_optical_depth_per_layer = 200
    dtaustar = np.minimum(
        dtaustar_ub, maximum_optical_depth_per_layer / kestar)
    # this is a dirty hack and causes problem with the irradiance profile calculation.
    # This is reason why we need to return the unbunded and the bunded dtaustar.
    # In practice, it is safe (but dirty) as a layer with optical >200 has a null transmittance

    taustar = np.cumsum(dtaustar, axis=1)

    return dtaustar_ub, dtaustar, taustar


def two_stream_matrix(layeralbedo, soilalbedo, kestar, dtaustar):
    """compute the matrix describing the boundary conditions at one wavelength.
    see doc Section 1.5

    :param layeralbedo: infinite albedo of each layers (no unit)
    :type layeralbedo: array
    :param soilalbedo: albedo of the bottom layer (no unit)
    :type soilalbedo: scalar
    :param kestar: delta-eddington AFEC (no unit)
    :type kestar: array
    :param dtaustar: optical depth (no unit)
    :type dtaustar: array

    :returns: tri-diagonal of the boundary matrix
"""

    nlyr = len(dtaustar)

    f_diag = np.exp(- kestar * dtaustar)

    Dm = np.zeros(2 * nlyr)
    Dm[0:-2:2] = (1 - layeralbedo[0:-1] * layeralbedo[1:]) * f_diag[0:-1]
    Dm[1:-1:2] = (1 / layeralbedo[0:-1] - layeralbedo[0:-1]) / f_diag[0:-1]

    D = np.zeros(2 * nlyr)
    D[1:-2:2] = (1 - layeralbedo[1:] / layeralbedo[0:-1]) / f_diag[0:-1]
    D[2:-1:2] = (layeralbedo[0:-1] - layeralbedo[1:])

    Dp = np.zeros(2 * nlyr)
    Dp[2:-1:2] = (layeralbedo[1:] * layeralbedo[1:] - 1)
    Dp[3::2] = (layeralbedo[0:-1] - 1. / layeralbedo[1:])

    # Bottom and top layer
    Dp[1] = 1
    D[0] = 1
    Dm[-2] = (layeralbedo[-1] - soilalbedo) * f_diag[-1]
    D[-1] = (1. / layeralbedo[-1] - soilalbedo) / f_diag[-1]

    d = np.array([Dp, D, Dm])

    return d


def Gp_Gm_vectors(ssalb, kestar, g, mu):
    """return Gp and Gm vectors at one wavelength

    :param ssalb: single scattering albedo of each layer (no unit)
    :type ssalb: array
    :param kestar: delta-eddington AFEC (no unit)
    :type kestar: array
    :param g: asymmetry factor  of each layer (no unit)
    :type g: array
    :param mu: cosine of the incident angle

    :returns: Gp and Gm
"""
    g_star = g / (1 + g)
    ssalb_star = ssalb * (1 - g**2) / (1 - g**2 * ssalb)

    gamma1 = 0.25 * (7 - ssalb_star * (4 + 3 * g_star))
    gamma2 = -0.25 * (1 - ssalb_star * (4 - 3 * g_star))

    gamma3 = 0.25 * (2 - 3 * g_star * mu)
    gamma4 = 0.25 * (2 + 3 * g_star * mu)

    G = mu**2 * ssalb_star / ((kestar * mu)**2 - 1)
    Gp = G * ((gamma1 - 1 / mu) * gamma3 + gamma2 * gamma4)
    Gm = G * ((gamma1 + 1 / mu) * gamma4 + gamma2 * gamma3)

    return Gp, Gm


def two_stream_vector(layeralbedo, soilalbedo, dtaustar, taustar, Gm, Gp, mu):
    """compute the vector for the boundary conditions
    see doc Section 1.5

    :param layeralbedo: albedo of the layer if it was infinite
    :type layeralbedo: array
    :param soilalbedo: albedo of the bottom layer
    :type soilalbedo: scalar
    :param dtaustar: optical depth
    :type dtaustar: array
    :param taustar: optical depth
    :type taustar: array
    :param Gm: coefficients Gm calculated for each layer
    :type Gm: array
    :param Gp: coefficients Gm calculated for each layer
    :type Gp: array
    :param mu: cosine of the incidence angle
    :type mu: array

    :returns: vector V
"""

    nlyr = len(taustar)
    vect = np.zeros(2 * nlyr)

    vect[0] = -Gm[0]

    dGp = np.diff(Gp)
    dGm = np.diff(Gm)

    vect[1:-2:2] = (dGm - layeralbedo[1:] * dGp) * np.exp(-taustar[0:-1] / mu)

    vect[2:-1:2] = (dGp - layeralbedo[0:-1] * dGm) * np.exp(-taustar[0:-1] / mu)

    vect[-1] = (soilalbedo * (Gm[-1] + mu) - Gp[-1]) * np.exp(-taustar[-1] / mu)

    return vect


def solves_two_stream(dmatrix, vect, layer, albedo):
    """solve the two stream linear system for vect.
    see doc Section 1.5

    :param dmatrix: two-stream matrix  M
    :type dmatrix: matrix
    :param vect: two-stream vector V
    :type vect: array or None
    :param layeralbedo: albedo of the layers if it was infinite
    :type layeralbedo: array

    :returns: vector X, unpacked solution vectors for coefficients A and B
"""
    # solve the two stream linear system

    solution0 = solve_banded((1, 1), dmatrix, vect)

    solution0 = solution0.squeeze()

    solution_A = solution0[:-1:2]
    solution_B = solution0[1::2]
    solution_C = solution_A * albedo
    solution_D = solution_B / albedo

    # TODO: use an object "Solution"
    return solution_A, solution_B, solution_C, solution_D


def solves_two_stream2(dmatrix, vect1, vect2, layeralbedo):
    """solve the two stream linear system for vect1 and vect2.

    :param dmatrix: two-stream matrix  M
    :type dmatrix: matrix
    :param vect1: two-stream vector V
    :type vect1: array or None
    :param vect2: two-stream vector V (2 vectors are used when ther is diffuse AND direct incident light)
    :type vect2: array or None


    :param layeralbedo: semi-infinite albedo of each layer
    :returns: vectors X, unpacked solution vectors
"""

    def unpack_solution(solution0):

        solution0 = solution0.squeeze()

        solution_A = solution0[:-1:2]
        solution_B = solution0[1::2]
        solution_C = solution_A * layeralbedo
        solution_D = solution_B / layeralbedo

        solution = solution_A, solution_B, solution_C, solution_D
        return solution

    if vect1 is None:
        assert vect2 is not None
        solution0 = solve_banded((1, 1), dmatrix, vect2)

        return None, unpack_solution(solution0)

    elif vect2 is None:
        assert vect1 is not None
        solution0 = solve_banded((1, 1), dmatrix, vect1)
        return unpack_solution(solution0), None

    else:
        vect = np.vstack((vect1, vect2)).T

        solution0 = solve_banded((1, 1), dmatrix, vect)
        return unpack_solution(solution0[:, 0]), unpack_solution(solution0[:, 1])


def snowpack_albedo(solutions, Gp, mu):
    """compute the albedo of the snowpack at one wavelength

    :param solutions: coefficients A et B pour chaque couche qui permttent calcul analytique des flux dans tous le manteau
    :type solutions: array
    :param Gp: coefficients Gm calculated for each layer
    :type Gp: array
    :param mu: cosine of the incident angle of intensity
    :ype mu:

    :returns: albedo at one wavelength (W/m^2)
"""

    if solutions is None:
        return 0

    solution_C = solutions[2]
    solution_D = solutions[3]

    albedo = (solution_C[0] + solution_D[0] + Gp[0]) / mu

    return albedo


def energy_profile(solutions, kestar, dtaustar, taustar, Gm, Gp, mu):
    """compute energy absorption for each layer at one wavelength

    :param solutions: coefficients A et B pour chaque couche qui permttent calcul analytique des flux dans tous le manteau
    :type solutions: array
    :param kestar: delta-eddington AFEC de chaque couche (no unit)
    :type kestar: array
    :param dtaustar: optical depth of each layer
    :type dtaustar: array
    :param taustar: cumulated optical depth at each interface
    :type taustar: array
    :param Gm: coefficients Gm calculated for each layer
    :type Gm: array
    :param Gp: coefficients Gm calculated for each layer
    :type Gp: array
    :param mu: cosine of the incident angle of intensity
    :type mu: scalar

    :returns: energy absorbed by each layer (W/m^2)
"""

    # compute energy absoprtion profile

    if solutions is None:
        return 0

    solution_A, solution_B, solution_C, solution_D = solutions

    eprofile = np.zeros_like(taustar)

    # surface Layer
    eprofile[0] = (mu - (solution_C[0] + solution_D[0] + Gp[0])) + \
                  ((solution_C[0] * np.exp(-kestar[0] * dtaustar[0]) +
                    solution_D[0] * np.exp(kestar[0] * dtaustar[0]) +
                    Gp[0] * np.exp(-taustar[0] / mu)) -
                   (solution_A[0] * np.exp(-kestar[0] * dtaustar[0]) +
                    solution_B[0] * np.exp(kestar[0] * dtaustar[0]) +
                    Gm[0] * np.exp(-taustar[0] / mu) + mu * np.exp(-taustar[0] / mu)))

    dexp = np.exp(-taustar[1:] / mu) - np.exp(- taustar[0:-1] / mu)

    expp = np.exp(kestar[1:] * dtaustar[1:])
    expm = np.exp(-kestar[1:] * dtaustar[1:])
    fdu = solution_C[1:] * (expm - 1) + solution_D[1:] * (expp - 1) + Gp[1:] * dexp
    fdd = solution_A[1:] * (expm - 1) + solution_B[1:] * \
        (expp - 1) + (Gm[1:] + mu) * dexp

    eprofile[1:] = fdu - fdd

    return eprofile


def soil_absorption(solutions, kestar, dtaustar, taustar, Gm, mu, soilalbedo):
    """compute the energy absorbed by the soil at one wavelength

    :param solutions: coefficients A et B pour chaque couche qui permttent calcul analytique des flux dans tous le manteau
    :type solutions: array
    :param kestar: delta-eddington AFEC de chaque couche (no unit)
    :type kestar: array
    :param dtaustar: optical depth of each layer
    :type dtaustar: array
    :param taustar: cumulated optical depth at each interface
    :type taustar: array
    :param Gm: coefficients Gm calculated for each layer
    :type Gm: array
    :param mu: cosine of the incident angle of intensity
    :type mu: scalar
    :param soilalbedo: soil albedo at that wavelength (no unit)
    :type soilalbedo: scalar

    :returns: energy absorbed by the soil at one wavelength (W/m^2)
"""

    if solutions is None:
        return 0

    solution_A = solutions[0]
    solution_B = solutions[1]

    # Soil absorption
    return (1 - soilalbedo) * (solution_A[-1] * np.exp(-kestar[-1] * dtaustar[-1]) +
                               solution_B[-1] * np.exp(kestar[-1] * dtaustar[-1]) +
                               (Gm[-1] + mu) * np.exp(-taustar[-1] / mu))


def estimate_effective_layer_number(wavelength, kestar, dtaustar):
    """estimate the number of layers to take into account at each wavelength

    :param wavelength: wavelength in m
    :type wavelength: array
    :param kestar: delta-eddington AFEC
    :type kestar: array
    :param dtaustar: optical depth of each layer
    :type dtaustar: array

    :returns: number of layers to consider for each wavelength
"""
    tau = np.cumsum(kestar * dtaustar, axis=1)
    taumax = 30.0  # optical depth from which the absorbed energy is negligible

    nlyrmax = np.empty_like(wavelength, dtype=np.int32)
    for i in range(len(nlyrmax)):
        # +1 added compared with Quentin's code
        nlyrmax[i] = np.searchsorted(tau[i, :], taumax, side='right') + 1

    return nlyrmax


def soa(x, i):
    # allow scalar or array/list

    if hasattr(x, "__iter__") and not isinstance(x, dict):
        return x[i]
    else:
        return x


def tartes(wavelength, SSA, density, thickness=None,
           g0=default_g0, B0=default_B0, y0=0.728, W0=0.0611,
           impurities=0, impurities_type=Soot,
           soilalbedo=0.0, dir_frac=0.0, totflux=1.0, mudir=0, return_dir_diff=False,
           compute_absorption=False, compute_irradiance_profiles=False, compute_actinic_profile=False,
           z=None, refrac_index=None):
    """compute spectral albedo, and optionally the absorption in each layer and in the soil, the downwelling and upwelling irradiance profiles and the actinic flux from the physical properties of the snowpack and the incidence flux conditions.

:param wavelength: wavelength (m)
:type wavelength: array or scalar
:param SSA: snow specific surface area (m^2/kg)
:type SSA: array or scalar
:param density: snow density (kg/m^3)
:type density: array or scalar
:param thickness: thickness of the layers (m)
:type thickness: array or scalar
:param g0: asymmetry parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 0.86. g0 can be scalar (constant in the snowpack) or an array like the SSA.
:type g0: array or scalar
:param B0: absorption enhancement parameter of snow grains at nr=1.3 and at non absorbing wavelengths (no unit). The default value is 1.6, taken from Libois et al. 2014. B0 can be scalar (constant in the snowpack) or an array like the SSA.
:type B0: array or scalar
:param y0: Value of y of snow grains at nr=1.3 (no unit).  See Eqs 72 and 75 in the "science" doc for the default value. y0 can be scalar (same for all layers) or an array like the SSA.
:type y0: array or scalar
:param W0: Value of W of snow grains at nr=1.3 (no unit). See Eqs 72 and 75 in the "science" doc for the default value. W0 can be a scalar (same for all layers) or an array like the SSA.
:type W0: array or scalar
:param impurities: impurities concentration (g/g/) in each layer. It is either a constant or an array with size equal to the number of layers. The array is 1-d if only one type of impurities is used and 2-d otherwise.
:type impurities: array or scalar
:param impurities_type: specify the type of impurity. By defaut it is "soot". Otherwise it should be a class (or an instance) defining the density and the imaginary part of the refractive index like the Soot class (see tartes.impurities for possible choices and to add new impurities types). It can also be a list of classes if several impurities type are present in the snowpack. In this case, the impurities parameter must be a 2-d array.
:type impurities_type: object or list of object
:param soilalbedo: albedo of the bottom layer (no unit). soilalbedo can be a scalar or an array like wavelength.
:type soilalbedo: scalar or array
:param dir_frac: fraction of directional flux over the total flux (the default is dir_frac = 0 meaning 100% diffuse incident flux) at every wavelength
:type dir_frac: array
:param totflux: total spectral incident flux (direct+diffuse) (W/m^2)
:type totflux: array
:param mudir: cosine of the incident angle of direct light
:type mudir: scalar
:param return_dir_diff: if True return the direct and diffuse components of the albedo. In this case, dir_frac and totflux are not used.
:type return_dir_diff: boolean
:param compute_absorption: if True compute the absorption profile and the absorption in the soil
:type compute_absorption: boolean
:param compute_irradiance_profiles: if True compute the profiles of up- and down-welling irradiance at depths z
:type compute_actinic_profile: boolean
:param compute_actinic_profile: if True compute the profile of actinic flux at depths z
:type compute_irradiance_profiles: boolean
:param z: depth at which the irradiance is calculed. It is used only if compute_irradiance_profile is activated.
:type z: array
:param refrac_index: real and imaginary parts of the refractive index for each wavelength or a string "p2016" for Picard et al. 2016 or w2008 for Warren and Brandt 2008 or w1995 for Warren, 1984 modified as noted on Warren web site.
:type refrac_index: string or tuple of two arrays

:returns: spectral albedo, and optionaly absorption by layer (note the bottom layer correspond to the absorption by the soil) and optionnaly the profile of irradiance: downwelling, upwelling
"""

    # the diffuse incident flux is treated as direct flux at incident angle 53°
    mudiff = np.cos(np.deg2rad(53.))

    # convert SSA, density and thickness to array if necessary
    SSA = np.atleast_1d(SSA)
    if density is None:
        if thickness is not None:
            raise Exception(
                "no density argument is only allowed for semi-infinite snowpacks (thickness=None)")
        density = 300  # any value is good

    density = np.atleast_1d(density)
    if thickness is None:
        thickness = 1e9  # chosing a value should never been a problem...
    thickness = np.atleast_1d(thickness)

    wavelength = np.atleast_1d(wavelength)
    N = len(wavelength)
    nlyr = len(SSA)

    if return_dir_diff:
        albedo_dir, albedo_diff = np.zeros(N)
    else:
        albedo = np.zeros(N)

    if compute_absorption:
        energyprofile = np.zeros((N, nlyr + 1))

    if compute_irradiance_profiles or compute_actinic_profile:
        nz = len(z)
        if compute_irradiance_profiles:
            down_irr_profile = np.zeros((N, nz))
            up_irr_profile = np.zeros((N, nz))
        if compute_actinic_profile:
            actinic_profile = np.zeros((N, nz))

        thickness_total = np.zeros(nlyr + 1)
        thickness_total[1:] = np.cumsum(thickness)
        # number of the layer (1,2...)
        nearest_layer = np.searchsorted(thickness_total, z, side="right")

    # intermediate variables
    ssalb = np.zeros((N, nlyr))
    g = np.zeros_like(ssalb)
    alb = np.zeros_like(ssalb)
    kestar = np.zeros_like(alb)

    # 1 compute optical properties for an array of wavelength
    if refrac_index is None or refrac_index == "w2008":
        # should be cached when the same wavelengths are used
        refrac_index = refice2008(wavelength)
    elif refrac_index == "w1995":
        # should be cached when the same wavelengths are used
        refrac_index = refice1995(wavelength)
    elif refrac_index == "p2016":
        # should be cached when the same wavelengths are used
        refrac_index = refice2016(wavelength)
    elif callable(refrac_index):
        # should be cached when the same wavelengths are used
        refrac_index = refrac_index(wavelength)

    for n in range(nlyr):
        ssalb[:, n], g[:, n] = single_scattering_optical_parameters(wavelength,
                                                                    refrac_index,
                                                                    SSA[n], density[n],
                                                                    soa(impurities, n),
                                                                    impurities_type,
                                                                    soa(g0, n), soa(y0, n), soa(W0, n), soa(B0, n))
        alb[:, n], kestar[:, n] = infinite_medium_optical_parameters(
            ssalb[:, n], g[:, n])

    # 2 computation on every wavelength and layer of the optical depth
    dtaustar_ub, dtaustar, taustar = taustar_vector(SSA, density, thickness, ssalb, g, kestar)

    # use to limit the computation at depth for highly absorbing wavelength. Seems to be inefficient in Python for a normal ~40 layers snowpack and 10nm resolution.
    nlyrmax = estimate_effective_layer_number(wavelength, kestar, dtaustar)

    # 3 solve the radiative transfer for each wavelength successively
    # TODO: convert this loop to parallal multi-core computing with joblib
    for i in range(0, N):

        # Number of layer required for the computation
        neff = min(nlyrmax[i], nlyr)
        if compute_irradiance_profiles:
            neff = max(neff, max(nearest_layer))

        layeralbedo_i = alb[i, :neff]
        kestar_i = kestar[i, :neff]
        soilalbedo_i = soa(soilalbedo, i)
        dir_frac_i = soa(dir_frac, i)
        totflux_i = soa(totflux, i)

        if return_dir_diff:
            diffrad_i = 1.
            dirrad_i = 1.
        else:
            if mudiff > 0:
                diffrad_i = (1.0 - dir_frac_i) * totflux_i / mudiff
            else:
                diffrad_i = 0

            if mudir > 0:
                dirrad_i = dir_frac_i * totflux_i / mudir
            else:
                dirrad_i = 0

        if dirrad_i > 0:
            Gpdir_i, Gmdir_i = Gp_Gm_vectors(
                ssalb[i, :neff], kestar_i, g[i, :neff], mudir)
        else:
            Gpdir_i = 0
            Gmdir_i = 0

        if diffrad_i > 0:
            Gpdiff_i, Gmdiff_i = Gp_Gm_vectors(
                ssalb[i, :neff], kestar_i, g[i, :neff], mudiff)
        else:
            Gpdiff_i = 0
            Gmdiff_i = 0

        # If the snowpack was truncated, the last layer thickness is increased and a high albedo is used for the soil
        if neff < nlyr:
            dtaustar[i, neff - 1] = 30. / kestar[i, neff - 1]
            soilalbedo_i = 1

        taustar_i = taustar[i, :neff]
        dtaustar_i = dtaustar[i, :neff]

        # compute the two-stream matrix
        d = two_stream_matrix(layeralbedo_i, soilalbedo_i,
                              kestar_i, dtaustar_i)

        # compute the vector for direct and diffuse intensities
        if dirrad_i > 0:
            vect_dir = two_stream_vector(layeralbedo_i, soilalbedo_i, dtaustar_i, taustar_i, Gmdir_i, Gpdir_i, mudir)
        else:
            vect_dir = None

        if diffrad_i > 0:
            vect_diff = two_stream_vector(layeralbedo_i, soilalbedo_i, dtaustar_i, taustar_i, Gmdiff_i, Gpdiff_i, mudiff)
        else:
            vect_diff = None

        solutions_dir, solutions_diff = solves_two_stream2(d, vect_dir, vect_diff, layeralbedo_i)

        # compute the albedo
        if (dirrad_i + diffrad_i) != 0:  # avoid infinite albedo
            adir = snowpack_albedo(solutions_dir, Gpdir_i, mudir)
            adiff = snowpack_albedo(solutions_diff, Gpdiff_i, mudiff)

            if return_dir_diff:
                albedo_dir[i] = adir
                albedo_diff[i] = adiff
            else:
                albedo[i] = (adir * dirrad_i * mudir + adiff * diffrad_i *
                             mudiff) / (dirrad_i * mudir + diffrad_i * mudiff)
        else:
            albedo[i] = 0

        if compute_absorption:
            # compute the profile of absorbed energy

            energyprofile[i, :neff] = dirrad_i * energy_profile(solutions_dir, kestar_i, dtaustar_i, taustar_i, Gmdir_i, Gpdir_i, mudir) + \
                diffrad_i * energy_profile(solutions_diff, kestar_i,
                                           dtaustar_i, taustar_i, Gmdiff_i, Gpdiff_i, mudiff)

            # compute the energy absorbed by the soil

            energyprofile[i, -1] = dirrad_i * soil_absorption(solutions_dir, kestar_i,
                                                              dtaustar_i, taustar_i,
                                                              Gmdir_i, mudir, soilalbedo_i) + \
                                   diffrad_i * soil_absorption(solutions_diff, kestar_i,
                                                               dtaustar_i, taustar_i,
                                                               Gmdiff_i, mudiff, soilalbedo_i)

        if compute_irradiance_profiles or compute_actinic_profile:
            # compute the profile of downward intensity
            if solutions_dir is not None:
                A_dir, B_dir, C_dir, D_dir = solutions_dir
            if solutions_diff is not None:
                A_diff, B_diff, C_diff, D_diff = solutions_diff

            # it is probably possible to optimize this loop (-> array calculation)
            for nz0, z0 in enumerate(z):
                m = nearest_layer[nz0]
                dtaustar_z = (z0 - thickness_total[m - 1]) / (thickness_total[m] - thickness_total[m - 1]) * dtaustar_ub[i, m - 1]
                taustar_z = dtaustar_z
                if m > 1:
                    taustar_z += taustar_i[m - 2]

                expm = np.exp(-kestar_i[m - 1] * dtaustar_z)
                expp = np.exp(kestar_i[m - 1] * dtaustar_z)

                if compute_irradiance_profiles:
                    down_irr_profile[i, nz0] = 0
                    up_irr_profile[i, nz0] = 0

                    if dirrad_i > 0 and mudir > 0:
                        down_irr_profile[i, nz0] = dirrad_i * (A_dir[m - 1] * expm + B_dir[m - 1] * expp +
                                                               (Gmdir_i[m - 1] + mudir) * np.exp(-taustar_z / mudir))
                        up_irr_profile[i, nz0] = dirrad_i * (C_dir[m - 1] * expm + D_dir[m - 1] * expp +
                                                             Gpdir_i[m - 1] * np.exp(-taustar_z / mudir))
                    if diffrad_i > 0 and mudiff > 0:
                        down_irr_profile[i, nz0] += diffrad_i * (A_diff[m - 1] * expm + B_diff[m - 1] * expp +
                                                                 (Gmdiff_i[m - 1] + mudiff) * np.exp(-taustar_z / mudiff))
                        up_irr_profile[i, nz0] += diffrad_i * (C_diff[m - 1] * expm + D_diff[m - 1] * expp +
                                                               Gpdiff_i[m - 1] * np.exp(-taustar_z / mudiff))

                if compute_actinic_profile:

                    actinic_profile[i, nz0] = 0

                    if dirrad_i > 0 and mudir > 0:
                        actinic_profile[i, nz0] = dirrad_i * (2 * (A_dir[m - 1] + C_dir[m - 1]) * expm +
                                                              2 * (B_dir[m - 1] + D_dir[m - 1]) * expp +
                                                              (Gmdir_i[m - 1] + mudir) * np.exp(-taustar_z / mudir))
                    if diffrad_i > 0 and mudiff > 0:
                        actinic_profile[i, nz0] += diffrad_i * (2 * (A_diff[m - 1] + C_diff[m - 1]) * expm +
                                                                2 * (B_diff[m - 1] + D_diff[m - 1]) * expp +
                                                                (Gmdiff_i[m - 1] + mudiff) * np.exp(-taustar_z / mudiff))

    if not compute_absorption and not compute_irradiance_profiles and not compute_actinic_profile:
        if return_dir_diff:
            return albedo_dir.squeeze(), albedo_diff.squeeze()
        else:
            return albedo.squeeze()
    else:
        ret = [albedo.squeeze()]
        if compute_absorption:
            ret.append(energyprofile.squeeze())
        if compute_irradiance_profiles:
            ret.append(down_irr_profile)
            ret.append(up_irr_profile)
        if compute_actinic_profile:
            ret.append(actinic_profile)
        return tuple(ret)
