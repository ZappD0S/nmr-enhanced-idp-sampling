import numpy as np
from scipy.constants import mu_0, value, pi, h, hbar


class NMR_relaxation_rates:
    def __init__(self, amps, taus, B0_MHz, theta_deg=22):
        g15N = -2.7126 * 1e7 # rad * s^−1 * T^−1
        g1H = value("proton gyromag. ratio") # rad * s^−1 * T^−1
        rNH = 1.02 * 1e-10
        dCSA = -170.0 * 1e-6

        # g15N /= 2 * np.pi
        # g1H /= 2 * np.pi

        B0 = B0_MHz / value("proton gyromag. ratio in MHz/T")

        # omegaN = g15N / (2 * np.pi) * B0
        # omegaH = g1H / (2 * np.pi) * B0
        omegaN = g15N / (2 * np.pi) * B0
        omegaH = g1H / (2 * np.pi) * B0
        omega_diff = omegaH - omegaN
        omega_sum = omegaH + omegaN

        self._omegas = [omegaN, omegaH, omega_diff, omega_sum]

        amps = np.asarray(amps)
        taus = np.asarray(taus)

        # taus = np.multiply(taus, 1e-12)

        # ps to s
        taus = taus * 1e-12

        J = self.define_J(amps, taus)
        c = dCSA * omegaN / np.sqrt(3.0)
        # d = mu_0 * h * g1H * g15N / (8 * pi**2 * rNH**3)
        d = mu_0 * hbar * g1H * g15N / (4 * pi * rNH**3)

        # see https://doi.org/10.1021/jacs.6b02424
        theta = np.deg2rad(theta_deg)

        self.R1 = (
            d**2 / 4.0 * (6.0 * J(omega_sum) + J(omega_diff) + 3.0 * J(omegaN))
            + J(omegaN) * c**2
        )

        self.R2 = d**2 / 8.0 * (
            6.0 * J(omega_sum)
            + 6.0 * J(omegaH)
            + J(omega_diff)
            + 3.0 * J(omegaN)
            + 4.0 * J(0)
        ) + c**2 / 6 * (3.0 * J(omegaN) + 4.0 * J(0))

        self.NOE = 1.0 + d**2 / (4.0 * self.R1) * g1H / g15N * (
            6.0 * J(omega_sum) - J(omega_diff)
        )

        self.etaXY = (
            -np.sqrt(3.0)
            / 6.0
            * d
            * c
            * self.P2(np.cos(theta))
            * (3.0 * J(omegaN) + 4.0 * J(0))
        )

        self.etaZ = -np.sqrt(3.0) * d * c * self.P2(np.cos(theta)) * J(omegaN)

        self.rates = {
            "R1": self.R1,
            "R2": self.R2,
            "NOE": self.NOE,
            "etaXY": self.etaXY,
            "etaZ": self.etaZ,
        }

    # def define_J(self, amps, taus):
    #     def J(omega):
    #         # TODO: the 0.4 is 2 / 5
    #         # the 2 is always the same (it comes from the fourier transform)
    #         # but the 5 depends from the number of exponentials in the sum
    #         # also, we need to include the k
    #         all_terms = 0.4 * amps * taus / (1.0 + (omega * taus) ** 2)
    #         return np.sum(all_terms)

    #     return J

    def define_J(self, amps, taus):
        amps = amps[..., None, :]
        taus = taus[..., None, :]

        def J(omega):
            omega = np.asarray(omega)[..., None]
            all_terms = 2 / 5 * amps * taus / (1.0 + (omega * taus) ** 2)
            # all_terms = amps * taus / (1.0 + (omega * taus) ** 2)
            return np.squeeze(np.sum(all_terms, axis=-1))

        return J

    def P2(self, x):
        return (3.0 * x**2 - 1.0) / 2.0
