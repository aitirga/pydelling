"""
Subsurface Fracture Independent Solutions Helper (SubFISH)
"""
import logging

import mpmath as mp
import numpy as np
import scipy.special as scsp

logger = logging.getLogger(__name__)

class SubfishException(Exception):
    pass


def calculate_tang(tang_data):
    logger.info('Calculating Tang solution')
    # Load parameters
    b = tang_data["b"]
    theta = tang_data["poros"]
    tau = tang_data["tort"]
    alpha = tang_data["alpha"]
    D_star = tang_data["Dw"]
    my_lambda = tang_data["lambda"]
    R_prime = tang_data["Rm"]
    R = tang_data["Rf"]
    v = tang_data["v"]
    l = tang_data["l"]

    min_time = float(tang_data["min_time"])
    max_time = float(tang_data["max_time"])
    point_num = int(tang_data["point_num"])
    times = np.geomspace(min_time, max_time, point_num)
    solution = np.zeros(point_num)

    # Derived parameters
    D = alpha * v + D_star
    D_prime = tau * D_star
    nu = v / (2 * D)
    my_beta = np.sqrt(4 * R * D / v ** 2)
    A = b * R / (theta * np.sqrt(R_prime * D_prime))

    A_tilde = l * theta * np.sqrt(R_prime * D_prime) / (v * b)
    B_tilde = R * l / v

    def to_invert(s):
        return mp.exp(nu * l) * mp.exp(
            -nu * l * mp.sqrt(1 + my_beta ** 2 * (mp.sqrt(s + my_lambda) / A + s + my_lambda)))

    def solution_without_disp_insta(t):
        if t <= B_tilde:
            return 0

        exp1 = np.exp(-A_tilde ** 2 / (4 * (t - B_tilde)))
        exp2 = np.exp(-my_lambda * t)
        return A_tilde / (2 * np.sqrt(np.pi) * (t - B_tilde) ** (3 / 2)) * exp1 * exp2

    def solution_without_disp_const(t):
        if t <= B_tilde:
            return 0

        inside_erfc1 = A_tilde / (2 * np.sqrt(t - B_tilde)) - np.sqrt(my_lambda * (t - B_tilde))
        term1 = np.exp(-np.sqrt(my_lambda) * A_tilde) * scsp.erfc(inside_erfc1)

        inside_erfc2 = A_tilde / (2 * np.sqrt(t - B_tilde)) + np.sqrt(my_lambda * (t - B_tilde))
        term2 = np.exp(np.sqrt(my_lambda) * A_tilde) * scsp.erfc(inside_erfc2)

        return 0.5 * np.exp(-B_tilde * my_lambda) * (term1 + term2)

    for step in range(point_num):
        t = times[step]

        if tang_data.get("neglect_long_disp", False):
            # Use analytical expression
            if tang_data["injection_type"] == "instantaneous":
                solution[step] = solution_without_disp_insta(t)
            elif tang_data["injection_type"] == "constant":
                solution[step] = solution_without_disp_const(t)
            else:
                raise SubfishException("Unknown injection_type '" + tang_data["injection_type"] + "'")

        else:
            if tang_data["injection_type"] == "instantaneous":
                solution[step] = mp.invertlaplace(to_invert, t)
            elif tang_data["injection_type"] == "constant":
                solution[step] = mp.invertlaplace(lambda s: to_invert(s) / s, t)
            else:
                raise SubfishException("Unknown injection_type '" + tang_data["injection_type"] + "'")

    return [times, solution]