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


# The impurities are defined either in terms of refractive index or in terms of absorption coefficient
# Both is forbidden !

import numpy as np
import math


class Soot(object):
    """class defining soot"""

    density = 1800.0

    @classmethod
    def refractive_index_imag(cls, wavelength):
        """return the imaginary part of the refracive index (= absorption) for soot."""
        # index from Chang (1990)
        # wl_um=1e6*wavelength

        # index_soot_real=1.811+0.1263*log(wl_um)+0.027*log(wl_um)**2+0.0417*log(wl_um)**3
        # index_soot_im=0.5821+0.1213*log(wl_um)+0.2309*log(wl_um)**2-0.01*log(wl_um)**3

        #m_soot = index_soot_real -1j * index_soot_im

        m_soot = 1.95 - 0.79j
        n = (m_soot**2 - 1) / (m_soot**2 + 2)  # absorption cross section of small particles (Bohren and Huffman, 1983)

        return n.imag


class HULIS(object):

    density = 1500.0

    @classmethod
    def refractive_index_imag(cls, wavelength):
        """return the imaginary part of the refracive index (= absorption) for HULIS."""

        # HULIS from Hoffer (2006)
        wl_um = 1e6 * wavelength
        m_hulis = 1.67 - 8e17 * 1j * (wl_um * 1e3)**(-7.0639) * 1e3 * cls.density * wl_um * 1e-6 / (4 * math.pi)
        n = (m_hulis**2 - 1) / (m_hulis**2 + 2)

        return n.imag


class CaponiDust(object):

    # based on
    # https://www.atmos-chem-phys.net/17/7175/2017/acp-17-7175-2017.pdf
    # on the basis of this, we compute absorption coefficient k_abs as
    # k_abs(lambda) = MAE(lambda) * concentration
    # with basb = absorption coeff
    #
    # and MAE(lambda) = MAE(400nm) (lambda/400nm)**AAE

    # Table of desertic dust from https://www.atmos-chem-phys.net/17/7175/2017/acp-17-7175-2017.pdf

    MAE400_AAE = {
        "PM2_5": {
            # Sahara
            'libya': (110., 4.1),

            'marocco': (90., 2.6),

            'algeria': (73., 2.8),
            # Sahel
            'mali': (630., 3.4),
            # Middle East
            'saudi_arabia': (130., 4.5),

            'kuwait': (310., 3.4),

            # Southern Africa
            'namibia': (135., 5.1),
            # Eastern Asia
            'china': (180., 3.2),
            # Australia
            'australia': (293., 2.9),
        },
        "PM10": {
            # Sahara
            'libya': (77., 3.2),

            'algeria': (82., 2.5),
            # Sahel
            'bodele': (27., 3.3),
            # Middle East
            'saudi_arabia': (81., 4.1),

            # Southern Africa
            'namibia': (50., 4.7),
            # Eastern Asia
            'china': (59., 3.),
            # North America
            'arizona': (103., 3.1),
            # South America
            'patagonia': (83., 2.9),
            # Australia
            'australia': (124., 2.9),
        }
    }

    @classmethod
    def location_list(cls, pm=None):

        loc = list()
        for _ in cls.MAE400_AAE:
            if pm is None or pm == _:
                loc += cls.MAE400_AAE[pm].keys()
        return set(loc)

    @classmethod
    def size_list(cls, location=None):

        pm = set()

        for s in cls.MAE400_AAE:
            if (location is None) or (location in cls.MAE400_AAE[s]):
                pm.add(s)
        return pm

    def __init__(self, location, size):
        self.location = location
        self.size = size
        if size not in self.MAE400_AAE:
            raise ValueError("The size is not available")
        if location not in self.MAE400_AAE[size]:
            raise ValueError("The location is not available for the given size")

    def MAE(self, wavelength):
        """return the Mass absorption efficiency of the particles in m2/kg"""

        MAE400, AAE = self.MAE400_AAE[self.size][self.location]

        MAE = MAE400 * (wavelength / 400e-9)**(-AAE)

        return MAE


class CrocusDust(object):
    """class defining dust imaginary refractive index from Muller et al., 2011 (one of the higher bound of dust absorption found in the literrature) and 
    Skiles et al.,2014 (one of lower bound of dust absorption found in the literrature). Muller et al., 2011 is default
    François Tuzet, Marie Dumont, June 2018"""

    density = 2600.0

    wavelength_interp_dust = [299e-3, 350e-3, 400e-3, 450e-3, 500e-3, 550e-3, 600e-3, 650e-3, 700e-3, 750e-3,
                              800e-3, 900e-3, 1000e-3, 1100e-3, 1200e-3, 1300e-3, 1400e-3, 1500e-3, 1600e-3, 1700e-3, 2501e-3]
    index_dust = {'muller2011': [0.038, 0.0312, 0.0193, 0.011, 0.0076, 0.0048, 0.003, 0.0025, 0.0021,
                                 0.002, 0.0018, 0.0017, 0.0016, 0.0016, 0.0016, 0.0015, 0.0015, 0.0015, 0.0014, 0.0014, 0.0014],
                  'skiles2014': [0.0019, 0.0018, 0.0016, 0.0013, 0.0011, 0.0009, 0.0008, 0.0007, 0.00067, 0.00064,
                                 0.00062, 0.00063, 0.00059, 0.00057, 0.00054, 0.00052, 0.00055, 0.00052, 0.0005, 0.00048, 0.00048]
                  }

    def __init__(self, formulation='muller2011'):

        if formulation not in self.index_dust:
            raise ValueError("Refractive index not available")

        self.formulation = formulation

    def refractive_index_imag(self, wavelength):
        """return the absorption cross section of small particles (Bohren and Huffman, 1983) for a given type of dust

        :param wavelength: wavelength (in m)
        :param formulation: by default use "muller2011" but "skiles2014" is also available.
        : """

        wl_um = 1e6 * wavelength
        index_dust_real = 1.53  # real part of the refractive index
        index_dust_im = np.exp(np.interp(np.log(wl_um),
                                         np.log(self.wavelength_interp_dust),
                                         np.log(self.index_dust[self.formulation])))
        m_dust = index_dust_real - 1j * index_dust_im

        n = (m_dust**2 - 1) / (m_dust**2 + 2)

        return n.imag
