
import numpy as np
from tartes.impurities import Soot, CrocusDust


def test_soot():

    ref_imag = Soot.refractive_index_imag(900e-9)
    np.testing.assert_allclose(ref_imag, -0.2545692778211603)


def test_crocusdust():

    muller2011 = CrocusDust(formulation='muller2011')
    ref_imag = muller2011.refractive_index_imag(900e-9)
    np.testing.assert_allclose(ref_imag, -0.000828193837574707)
