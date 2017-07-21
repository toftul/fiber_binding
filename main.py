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
alpha0 = 1.

hight = 1. 

n_times_k0 = 1

tmax = 1.
dt = 1.
dz = 1.

# r = {rho=x, y, z}
dip_r = np.zeros([2, 3])
dip_r[0] = np.array([hight, 0., 0.])
dip_r[1] = np.array([hight, 0., 5.])
dip_v = np.zeros([2, 3])
dip_mu = np.zeros([2, 3], dtype=complex)





a_charastic = a_charastic_nm * 1e-9  # [m]
m_charastic = 4 / 3 * np.pi * a_charastic ** 3 * density_of_particle  # [kg]
step_num = int(tmax / dt)
time_space = np.linspace(0, tmax, step_num + 1)




z_for_force_calc = np.linspace(0, 30)

# incident wave vector
def k1(wl):
    return(2 * np.pi / wl)

def G0zz(x1, y1, z1, x2, y2, z2, wl):
    Rmod = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    
    kR = k1(wl) * Rmod
    EXPikR4piR = np.exp(1j * kR) / (4. * np.pi * Rmod)
    A = 1 + (1j * kR - 1) / kR**2
    B = (3 - 3j * kR - kR**2) / (kR**2 * Rmod**2) \
                * (z1 - z2)**2 / Rmod**2
    Gzz = EXPikR4piR * (A + B)
        
    return(Gzz)


#
#   1 rho                 
#   |                  |------> E0
#   |                  |  
#   |    /-\           |      incident wave
#   |   | * |          v k1
#   |    \-/
#   |
#  ##############################
#  ~                            ~
#  ~           FIBER            ~ -----> z
#  ~                            ~
#  ##############################
#
#
# let particles has the same hight -> rho = rho'

# waveguide wave vector 
def k1_wg():
    return(0)
def kz_wg():
    return(0)
def krho1_wg():
    return(0)
def krho2_wg():
    return(0)
def A():
    return(0)
def B():
    return(0)
def C():
    return(0)
def A_prime(wl):
    return(0)
def B_prime(wl):
    return(0)
def C_prime(wl):
    return(0)
def F(wl):
    return(0)
def P11NN():
    return(0)
def Gszz(rho, rho_prime, dz, wl=wave_length):
    G = -0.5 * P11NN() * krho1_wg()**2 / (k1_wg()**2) \
        * sp.hankel1(0, krho1_wg() * rho) * sp.hankel1(0, krho1_wg() * rho_prime) \
        * np.exp(1j * kz_wg() * np.abs(dz))
    return(G)

# mu_12 = alpha_eff * E0
# ( 1,2 -- only if rho1 = rho2 )
def alpha_eff(wl):
    # Gs(r1, r2) = Gs(r2,r1)
    GGG = Gszz(dip_r[0, 0], dip_r[0, 0], 0.) + \
          Gszz(dip_r[0, 0], dip_r[1, 0], dip_r[0, 2] - dip_r[1, 2]) + \
          G0zz(dip_r[0], dip_r[1], wl)
    return(alpha0 / (1 - alpha0 * k1(wl)**2 / const.epsilon0 * GGG))
    

# z may be a numpy array! Should be fast
# considered that d/dz mu = 0
def force(rho1, rho2, z, wl=wave_length):
    al_eff = alpha_eff(wl)
    Fz = 0.5 * al_eff * al_eff.conjugate() * \
         E0_charastic * E0_charastic.conjugate() / const.epsilon0
         
    z_plus = z + dz
    z_minus = z - dz
    
    G_plus =  Gszz(rho1, rho2, z_plus) + \
              G0zz(rho1, 0., 0., rho2, 0., z_plus, wl)
              
    G_minus = Gszz(rho1, rho2, z_minus) + \
              G0zz(rho1, 0., 0., rho2, 0., z_minus, wl)
              
    RE = k1(wl)**2 * (G_plus - G_minus) * 0.5 / dz
    
    return(Fz * RE.real)


    
