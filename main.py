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


#                
#                   1 rho                 
#                   |                  |------> E0
#                   |                  |  
#                   |    /-\           |      incident wave
#                   |   | * |          v k1
#                   |    \-/
#                   |
#  1 krho          ##############################
#  |               ~                            ~
#  ------>         ~           FIBER            ~ -----> z
#        kz        ~                            ~
#                  ##############################
#                
#                
#                 let particles has the same hight -> rho = rho'



# ## Creating variables
theta = 30 * np.pi / 180
E0_charastic = 1e4  # [V/m]
time_charastic = 10e-3  # [s]
a_charastic_nm = 3000.  # [nm]
# Silicon
density_of_particle = 2328.  # [kg / m^3]
wave_length = 350.  # [nm]
wave_length_wg = 550.  # [nm]
gamma = 0.  # phenomenological parameter
epsilon_fiber = 4.7  # Glass
epsilon_m = 1.  # Air

a_charastic = a_charastic_nm * 1e-9  # [m]

def epsilon_particle(wl):
    # Silicon
    return(11.64)

def alpha0(wl = wave_length):
    return(4 * np.pi * const.epsilon0 * a_charastic**3 * 
           (epsilon_particle(wl) - epsilon_m) / (epsilon_particle(wl) + 2 * epsilon_m))



# fiber radius
rho_c = 200e-9  # Wu and Tong, Nanophotonics 2, 407 (2013)

# must be int
n_mode = 1  # for HE11

hight = 5e-9  

tmax = 1e-3
dt = tmax / 1000
dz = a_charastic / 100

# r = {rho=x, y, z}
dip_r = np.zeros([2, 3])
dip_r[0] = np.array([hight, 0., 0.]) * 1e-9
dip_r[1] = np.array([hight, 0., 5*a_charastic_nm]) * 1e-9
dip_v = np.zeros([2, 3])
dip_mu = np.zeros([2, 3], dtype=complex)




m_charastic = 4 / 3 * np.pi * a_charastic ** 3 * density_of_particle  # [kg]
step_num = int(tmax / dt)
time_space = np.linspace(0, tmax, step_num + 1)




z_for_force_calc = np.linspace(0, 30)

# incident wave vector
def k1_inc(wl):
    return(2 * np.pi / wl)

def G0zz(x1, y1, z1, x2, y2, z2, wl):
    Rmod = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    
    kR = k1_inc(wl) * Rmod
    EXPikR4piR = np.exp(1j * kR) / (4. * np.pi * Rmod)
    A = 1 + (1j * kR - 1) / kR**2
    B = (3 - 3j * kR - kR**2) / (kR**2 * Rmod**2) \
                * (z1 - z2)**2 / Rmod**2
    Gzz = EXPikR4piR * (A + B)
        
    return(Gzz)


# waveguide wave vector 
def k1_wg():
    return(2 * np.pi / wave_length_wg)
def k2_wg():
    return(np.sqrt(epsilon_fiber) * k1_wg())

def kz_wg():
    return(k2_wg() * np.cos(theta))
def krho1_wg():
    return(k1_wg() * np.sin(theta))
def krho2_wg():
    return(k2_wg() * np.sin(theta))

def A_():
    return(1/krho2_wg()**2 - 1/krho1_wg()**2)
def B_():
    return(sp.jvp(n_mode, krho2_wg() * rho_c) / 
           (krho2_wg() * sp.jv(n_mode, krho2_wg() * rho_c)))
def C_():
    return(sp.h1vp(n_mode, krho1_wg() * rho_c) / 
           krho1_wg() * sp.hankel1(n_mode, krho1_wg() * rho_c))
def A_prime():
    return(2 * kz_wg() * (1/krho2_wg()**4 - 1/krho1_wg()**4))
def B_prime():
    kr2 = krho2_wg()
    kr = kr2 * rho_c
    Jn = sp.jv(n_mode, kr)
    Jnp = sp.jvp(n_mode, kr)
    Jnpp = sp.jvp(n_mode, kr, 2)
    
    B1 = - Jnpp * kz_wg() * rho_c / (kr2**2 * Jn)
    B2 = Jnp * kz_wg() * (Jn / kr2 + rho_c * Jnp) / (kr2 * Jn)**2
    return(B1 + B2)
def C_prime():
    kr1 = krho1_wg()
    kr = kr1 * rho_c
    Hn = sp.hankel1(n_mode, kr)
    Hnp = sp.h1vp(n_mode, kr)
    Hnpp = sp.h1vp(n_mode, kr, 2)
    
    C1 = - Hnpp * kz_wg() * rho_c / (kr1**2 * Hn)
    C2 = Hnp * kz_wg() * (Hn / kr1 + rho_c * Hnp) / (kr1 * Hn)**2
    return(C1 + C2)
def F_():
    A = A_()
    Ap = A_prime()
    B = B_()
    Bp = B_prime()
    C = C_()
    Cp = C_prime()
    kz = kz_wg()
    k1 = k1_wg()
    k2 = k2_wg()
    
    F = - 2 * A * kz * epsilon_fiber * (Ap * kz + A) + \
        ((Bp - Cp) * (k2**2 * B - k1**2 * C) + (B - C) * (k2**2 * Bp - k1**2 * Cp)) * rho_c**2
    
    return(F)
def P11NN():
    k1 = k1_wg()
    k2 = k2_wg()
    kr1 = krho1_wg()
    kr2 = krho2_wg()
    kr1rc = kr1 * rho_c
    kr2rc = kr2 * rho_c
    Jn1 = sp.jv(n_mode, kr1rc)
    Jn2 = sp.jv(n_mode, kr2rc)
    Jnp1 = sp.jvp(n_mode, kr1rc)
    Jnp2 = sp.jvp(n_mode, kr2rc)
    Hn1 = sp.hankel1(n_mode, kr1rc)
    Hnp1 = sp.h1vp(n_mode, kr1rc)
    F = F_()
    
    P = (1/kr2**2 - 1/kr1**2) * kz_wg()**2 * epsilon_fiber - \
        (Jnp2 / (kr2 * Jn2) - Hnp1 / (kr1 * Hn1)) *\
        (Jnp2 * k2**2 / (Jn2 * kr2) - Jnp1 * k1**2 / (Jn1 * kr1)) * rho_c**2
    
    return(Jn1 / (F * Hn1) * P)
def Gszz(rho, rho_prime, dz, wl = wave_length_wg):
    G = -0.5 * P11NN() * krho1_wg()**2 / (k1_wg()**2) * \
        sp.hankel1(0, krho1_wg()*rho) * sp.hankel1(0, krho1_wg()*rho_prime) * \
        np.exp(1j * kz_wg() * np.abs(dz))
    return(G)

# mu_12 = alpha_eff * E0
# ( 1,2 -- only if rho1 = rho2 )
def alpha_eff(rho1, rho2, z, wl):
    # Gs(r1, r2) = Gs(r2,r1)
    GGG = Gszz(rho1, rho2, 0, wl) + Gszz(rho1, rho2, z, wl) + \
          G0zz(rho1, 0., 0., rho2, 0., z, wl)
    return(alpha0(wl) / (1 - alpha0(wl) * k1_inc(wl)**2 / const.epsilon0 * GGG))
    

# z may be a numpy array! Should be fast
# considered that d/dz mu = 0
def force(rho1, rho2, z, wl=wave_length):
    al_eff = alpha_eff(rho1, rho2, z, wl)
    Fz = 0.5 * al_eff * al_eff.conjugate() * \
         E0_charastic * E0_charastic.conjugate() / const.epsilon0
         
    z_plus = z + dz
    z_minus = z - dz
    
    G_plus =  Gszz(rho1, rho2, z_plus) + \
              G0zz(rho1, 0., 0., rho2, 0., z_plus, wl)
              
    G_minus = Gszz(rho1, rho2, z_minus) + \
              G0zz(rho1, 0., 0., rho2, 0., z_minus, wl)
              
    RE = k1_inc(wl)**2 * (G_plus - G_minus) * 0.5 / dz
    
    return(Fz.real * RE.real)


def plot_F(rho1, rho2, z, wl=wave_length):
    f = force(rho1, rho2, z, wl)
    plt.plot(z*1e9, f*1e9)
    plt.xlabel(r'$\Delta z$, nm')
    plt.ylabel(r'$Fz$, nN')
    plt.grid()
    plt.show()
    
def plot_F_wl(rho1, rho2, z, wl):
    f = force(rho1, rho2, z, wl)
    plt.plot(wl*1e9, f*1e9)
    plt.xlabel(r'$\lambda$, nm')
    plt.ylabel(r'$Fz$, nN')
    plt.grid()
    plt.show()

plot_F(hight, hight, np.linspace(2*a_charastic, 100000*a_charastic, 300), 302.7e-9)

plot_F_wl(hight, hight, 30*a_charastic, np.linspace(302, 304, 400)*1e-9)