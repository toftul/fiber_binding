# -*- coding: utf-8 -*-
"""
Main project file
"""

# import standart libs
import numpy as np
#import scipy
#from scipy.integrate import ode, quad
#from scipy import interpolate
import matplotlib.pyplot as plt
#import progressbar
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
#                  ##############################
#                  ~                            ~
#                  ~           FIBER            ~ -----> z
#                  ~                            ~
#                  ##############################
#                
#                
#                 let particles has the same hight -> rho = rho'



# ## Creating variables
# theta = 30 * np.pi / 180
E0_charastic = 1e4  # [V/m]
#time_charastic = 10e-3  # [s]
a_charastic = 100e-9 # [m]
# Silicon
density_of_particle = 2328.  # [kg / m^3]
# wave_length = 350e-9  # [nm]
# wave_length_wg = 450e-9  # [nm]
gamma = 0.  # phenomenological parameter
epsilon_fiber = 4.7  # Glass
epsilon_m = 1.  # Air


def fd_dist(x, mu, T):
    return(1/(1+np.exp((x-mu)/T)))


def epsilon_particle(wl):
    # Silicon
    return(11.64)

def alpha0(wl):
    return(4 * np.pi * const.epsilon0 * a_charastic**3 * 
           (epsilon_particle(wl) - epsilon_m) / (epsilon_particle(wl) + 2 * epsilon_m))



# fiber radius
# Wu and Tong, Nanophotonics 2, 407 (2013)
rho_c = 200e-9  # [m]

# must be int
n_mode = 1  # for HE11

gap = 5e-9  # [m]

#tmax = 1e-3
#dt = tmax / 1000
dz = a_charastic / 100

# r = {rho=x, y, z}
#dip_r = np.zeros([2, 3])
#dip_r[0] = np.array([rho_c + gap, 0., 0.])
#dip_r[1] = np.array([rho_c + gap, 0., 5*a_charastic])
#dip_v = np.zeros([2, 3])
#dip_mu = np.zeros([2, 3], dtype=complex)


#m_charastic = 4 / 3 * np.pi * a_charastic ** 3 * density_of_particle  # [kg]

#step_num = int(tmax / dt)
#time_space = np.linspace(0, tmax, step_num + 1)


# incident wave vector
def k1_inc(wl):
    return(2 * np.pi / wl)

def G0zz(x1, y1, z1, x2, y2, z2, wl):
    Rmod = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    
    kR = k1_inc(wl) * Rmod
    EXPikR4piR = np.exp(1j * kR) / (4. * np.pi * Rmod)
    A = 1 + (1j * kR - 1) / kR**2
    B = (3 - 3j * kR - kR**2) / (kR**2) \
                * (z1 - z2)**2 / Rmod**2
    Gzz = EXPikR4piR * (A + B)
    return(Gzz)


# waveguide wave vector 
def k1_wg(wl):
    return(2 * np.pi / wl)
def k2_wg(wl):
    return(np.sqrt(epsilon_fiber) * k1_wg(wl))

omega_c = 2*np.pi / rho_c * 2 / (np.sqrt(epsilon_fiber) + 1) * const.c
def kz_wg(omega):
    koef = (1 - fd_dist(omega, omega_c, omega_c*0.2)) * (np.sqrt(epsilon_fiber) - 1) + 1
    return(koef / const.c * omega)

def krho1_wg(wl):
    return(np.sqrt(k1_wg(wl)**2 - kz_wg(2*np.pi*const.c/wl)))
def krho2_wg(wl):
    return(np.sqrt(k2_wg(wl)**2 - kz_wg(2*np.pi*const.c/wl)))

def A_(wl):
    return(1/krho2_wg(wl)**2 - 1/krho1_wg(wl)**2)
def B_(wl):
    return(sp.jvp(n_mode, krho2_wg(wl) * rho_c) / 
           (krho2_wg(wl) * sp.jv(n_mode, krho2_wg(wl) * rho_c)))
def C_(wl):
    return(sp.h1vp(n_mode, krho1_wg(wl) * rho_c) / 
           krho1_wg(wl) * sp.hankel1(n_mode, krho1_wg(wl) * rho_c))
def A_prime(wl):
    return(2 * kz_wg(wl) * (1/krho2_wg(wl)**4 - 1/krho1_wg(wl)**4))
def B_prime(wl):
    kr2 = krho2_wg(wl)
    kr = kr2 * rho_c
    Jn = sp.jv(n_mode, kr)
    Jnp = sp.jvp(n_mode, kr)
    Jnpp = sp.jvp(n_mode, kr, 2)
    
    B1 = - Jnpp * kz_wg(2*np.pi*const.c/wl) * rho_c / (kr2**2 * Jn)
    B2 = Jnp * kz_wg(2*np.pi*const.c/wl) * (Jn / kr2 + rho_c * Jnp) / (kr2 * Jn)**2
    return(B1 + B2)
def C_prime(wl):
    kr1 = krho1_wg(wl)
    kr = kr1 * rho_c
    Hn = sp.hankel1(n_mode, kr)
    Hnp = sp.h1vp(n_mode, kr)
    Hnpp = sp.h1vp(n_mode, kr, 2)
    
    C1 = - Hnpp * kz_wg(2*np.pi*const.c/wl) * rho_c / (kr1**2 * Hn)
    C2 = Hnp * kz_wg(2*np.pi*const.c/wl) * (Hn / kr1 + rho_c * Hnp) / (kr1 * Hn)**2
    return(C1 + C2)
def F_(wl):
    A = A_(wl)
    Ap = A_prime(wl)
    B = B_(wl)
    Bp = B_prime(wl)
    C = C_(wl)
    Cp = C_prime(wl)
    kz = kz_wg(2*np.pi*const.c/wl)
    k1 = k1_wg(wl)
    k2 = k2_wg(wl)
    
    F = - 2 * A * kz * epsilon_fiber * (Ap * kz + A) + \
        ((Bp - Cp) * (k2**2 * B - k1**2 * C) + (B - C) * (k2**2 * Bp - k1**2 * Cp)) * rho_c**2
    
    return(F)

def P11NN(wl):
    k1 = k1_wg(wl)
    k2 = k2_wg(wl)
    kr1 = krho1_wg(wl)
    kr2 = krho2_wg(wl)
    kr1rc = kr1 * rho_c
    kr2rc = kr2 * rho_c
    Jn1 = sp.jv(n_mode, kr1rc)
    Jn2 = sp.jv(n_mode, kr2rc)
    Jnp1 = sp.jvp(n_mode, kr1rc)
    Jnp2 = sp.jvp(n_mode, kr2rc)
    Hn1 = sp.hankel1(n_mode, kr1rc)
    Hnp1 = sp.h1vp(n_mode, kr1rc)
    F = F_(wl)
    
    P = (1/kr2**2 - 1/kr1**2)**2 * kz_wg(2*np.pi*const.c/wl)**2 * epsilon_fiber - \
        (Jnp2 / (kr2 * Jn2) - Hnp1 / (kr1 * Hn1)) *\
        (Jnp2 * k2**2 / (Jn2 * kr2) - Jnp1 * k1**2 / (Jn1 * kr1)) * rho_c**2
    
    return(Jn1 / (F * Hn1) * P)

def Gszz(rho, rho_prime, dz, wl):
    G = -0.5 * P11NN(wl) * krho1_wg(wl)**2 / (k1_wg(wl)**2) * \
        sp.hankel1(0, krho1_wg(wl)*rho) * sp.hankel1(0, krho1_wg(wl)*rho_prime) * \
        np.exp(1j * kz_wg(2*np.pi*const.c/wl) * np.abs(dz))
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
def force(rho1, rho2, z, wl):
    al_eff = alpha_eff(rho1, rho2, z, wl)
    Fz = 0.5 * al_eff * al_eff.conjugate() * \
         E0_charastic * E0_charastic.conjugate() / const.epsilon0
         
    z_plus = z + dz
    z_minus = z - dz
    
    G_plus =  Gszz(rho1, rho2, z_plus, wl) + \
              G0zz(rho1, 0., 0., rho2, 0., z_plus, wl)
              
    G_minus = Gszz(rho1, rho2, z_minus, wl) + \
              G0zz(rho1, 0., 0., rho2, 0., z_minus, wl)
              
    RE = k1_inc(wl)**2 * (G_plus - G_minus) * 0.5 / dz
    
    return(Fz.real * RE.real)

def plot_F_wl(rho1, rho2, z, wl):
    f = force(rho1, rho2, z, wl)
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7.24, 4.24))
    plt.plot(wl*1e9, f)
    #plt.ylim(-1e-20, 1e-20)
    plt.title(r'a = %.0f nm, $\rho_c$ = %.0f nm, $\Delta z$ = %.1f $\mu$m' % (a_charastic*1e9, rho_c*1e9, z*1e6), loc='right')
    plt.xlabel(r'$\lambda$, nm')
    plt.ylabel(r'$Fz$, N')
    plt.grid()
    plt.show()

def plot_F(rho1, rho2, z, wl):
    f = force(rho1, rho2, z, wl)
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7.24, 4.24))
    plt.plot(z*1e6, f)
    plt.title(r'a = %.0f nm, $\rho_c$ = %.0f nm, $\lambda$ = %.1f nm' % (a_charastic*1e9, rho_c*1e9, wl*1e9), loc='right')
    plt.xlabel(r'$\Delta z$, $\mu$m')
    plt.ylabel(r'$Fz$, N')
    #plt.ylim(-1e-18, 1e-18)
    plt.grid()
    plt.show()
    
def plot_alpha_z(rho1, rho2, z, wl):
    al = alpha_eff(rho1,rho2, z, wl)/alpha0(wl)
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7.24, 4.24))
    plt.plot(z*1e6, al)
    plt.legend()
    plt.title(r'a = %.0f nm, $\rho_c$ = %.0f nm, $\lambda$ = %.1f nm' % (a_charastic*1e9, rho_c*1e9, wl*1e9), loc='right')
    plt.xlabel(r'$\Delta z$, $\mu$m')
    plt.ylabel(r'$\alpha_{eff} / \alpha_0$')
    #plt.ylim(-1e-18, 1e-18)
    plt.grid()
    plt.show()    

def plot_G_z(rho1, rho2, z, wl):
    #al = alpha_eff(rho1,rho2, z, wl)
    al = G0zz(rho1, 0., 0., rho2, 0., z, wl)
    al2 = Gszz(rho1, rho2, z, wl)
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7.24, 4.24))
    plt.plot(z*1e6, al, label=r'$G_0$')
    plt.plot(z*1e6, al2, label=r'$G_s$')
    plt.legend()
    plt.title(r'a = %.0f nm, $\rho_c$ = %.0f nm, $\lambda$ = %.1f nm' % (a_charastic*1e9, rho_c*1e9, wl*1e9), loc='right')
    plt.xlabel(r'$\Delta z$, $\mu$m')
    plt.ylabel('Greens function')
    #plt.ylim(-1e-18, 1e-18)
    plt.grid()
    plt.show()
    
def plot_kz():
    plt.rcParams.update({'font.size': 14})
    lam = np.linspace(200, 4200, 200) * 1e-9
    omega = 2*np.pi*const.c/lam
    plt.xlabel(r'$k z$, m$^{-1}$')
    plt.ylabel(r'$\omega$, s$^{-1}$')
    plt.xlim(0, np.max(kz_wg(omega)) + np.max(kz_wg(omega))/20)
    plt.ylim(0, np.max(omega) + np.max(omega)/20)
    plt.plot(kz_wg(omega), omega)
    plt.plot(omega/const.c, omega, linestyle='--', color='black')
    plt.plot(np.sqrt(epsilon_fiber)*omega/const.c, omega, linestyle='--', color='black')
    plt.plot(np.ones(len(omega))*2*np.pi/rho_c, omega, linestyle='--', dashes=(5, 3), color='gray')
    plt.text(2*np.pi/rho_c*1.05, 0.2e16, r'$\frac{2\pi}{\rho_c}$')
    plt.show()

#plt.xkcd()        # on
#plt.rcdefaults()  # off
plot_F_wl(rho_c + gap, rho_c + gap, 30*a_charastic, np.linspace(400, 1300, 4000)*1e-9)
plot_F(rho_c + gap, rho_c + gap, np.linspace(0.2e-6, 1.5e-6, 30000), 600e-9)
plot_alpha_z(rho_c + gap, rho_c + gap, np.linspace(0.2e-6, 1.5e-6, 30000), 600e-9)
plot_G_z(rho_c + gap, rho_c + gap, np.linspace(0.2e-6, 1.5e-6, 30000), 600e-9)

