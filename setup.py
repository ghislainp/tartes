from distutils.core import setup, Command


class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


setup(
    name = "tartes",
    packages = ["tartes"],
    version = "1.3.3",
    description = "Two-stream radiative transfer in snow model",
    author = "Quentin Libois, Ghislain Picard",
    author_email = "quentin.libois@meteo.fr, ghislain.picard@univ-grenoble-alpes.fr",
    url = "http://gp.snow-physics.science/tartes/",
    keywords = ["radiative transfer","model","snow","optics"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        ],
    cmdclass = {'test': PyTest},
    long_description = """\
Two-stream radiative transfer in snow model
-------------------------------------------

TARTES is a fast and easy-to-use optical radiative transfer model used
to compute snow albedo (spectral and broadband) and energy absorption
profile. TARTES represents the snowpack as a stack of horizontal
homogeneous layers. Each layer is characterized by the snow specific
surface area (SSA), snow density, impurities amount and type, and two
parameters for the geometric grain shape: the asymetry factor g and
the absorption enhancement parameter B. The albedo of the bottom
interface can be prescribed. The model is fast and easy to use
compared to more elaborated models like DISORT - MIE (Stamnes et
al. 1988). It is based on the Kokhanovsky and Zege (2004) formalism
for weakly absorbing media to describe the single scattering
properties of each layers and the delta-eddington approximation to
solve the radiative transfer equation. Despite its simplicity, it is
accurate in the visible and near-infrared range for pristine snow as
well as snow containing impurities represented as Rayleigh scatterers
(their size is assumed much smaller than the wavelength) whose
refractive indices and concentrations can be prescribed.

TARTES is compatible with Python 2.7x and 3.4+.
"""
)
