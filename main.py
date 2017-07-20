# -*- coding: utf-8 -*-
"""
Main project file
"""

# import standart libs
import numpy as np
import scipy
from scipy.integrate import ode, quad
from scipy import interpolate
import matplotlib.pyplot as plt
import progressbar
import scipy.special as sp
# import my libs
import const
import prm
import extra



# ## Creating variables
# ## TO CHANGE VARIABLES GO TO prm.py file nearby!
E0_charastic = 1.  # [V/m]
time_charastic = 1.  # [s]
a_charastic_nm = 1.  # [nm]
density_of_particle = 1.  # [kg / m^3]
wave_length = 1.  # [nm]
gamma = 1.  # phenomenological parameter
epsilon_fiber = 1.
epsilon_particle = 1.
epsilon_m = 1.

hight = 1. 

n_times_k0 = 1

tmax = 1.
dt = 1.
dr = 1.

dip_r = np.zeros([2, 3])
dip_r[0] = np.array([0., 0., hight])
dip_r[1] = np.array([0., 5., hight])
dip_v = np.zeros([2, 3])
dip_mu = np.zeros([2, 3], dtype=complex)
dip_mass = np.ones(2) * m_charastic





a_charastic = a_charastic_nm * 1e-9  # [m]
m_charastic = 4 / 3 * np.pi * a_charastic ** 3 * density_of_particle  # [kg]
step_num = int(tmax / dt)
time_space = np.linspace(0, tmax, step_num + 1)




z_for_force_calc = np.linspace(0, 30)


def G0(Rmod, wl):
    result = np.zeros([3, 3])
    if Rmod != 0:
        kR = k0(wl) * Rmod
        EXPikR4piR = np.exp(1j * kR) / (4. * np.pi * Rmod)
        CONST1 = (1 + (1j * kR - 1) / kR**2)
        CONST2 = (3 - 3j * kR - kR**2) / (kR**2 * Rmod**2)
        # return Green function
        result = EXPikR4piR * (CONST1 * np.identity(3) +
                               CONST2 * np.outer(R, R))
    else:
        print("G0 ERROR! Rmod = 0 leads to singularity.")

    return result



#
#   1 rho                 
#   |                  |------> E0
#   |                  |  
#   |    /-\           |
#   |   | * |          v k1
#   |    \-/
#  ##############################-----> z
#  ~                            ~
#  ~            FIBER           ~
#  ~                            ~
#  ##############################
#
#
# for normal incidence kz = 0
def A_prime(wl):
    return(0)
def B_prime(wl):
    return(0)
def C_prime(wl):
    return(0)
def F(wl):
    return(0)
def P11NN(wl):
    return(0)
def Gszz(rho, rho_prime, dz, wl=wave_length):
    G = -0.5 * P11NN(wl) * krho1(wl)**2 / (k1(wl)**2) * \
        sp.hankel1(0, krho1 * rho) * sp.hankel1(0, krho1 * rho_prime)


def mu_solver():
    


def force(z, wl=wave_length):
    
    
