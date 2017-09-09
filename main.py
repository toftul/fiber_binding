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
import matplotlib as mpl
#import progressbar
import scipy.special as sp
# import my libs
import const


#        
#           1 rho                 
#           |                  |------> E0
#           |                  |  
#           |    /-\           |      incident wave
#           |   | * |          v k1
#           |    \-/
#           |
#          ##############################
#          ~                            ~
#          ~   FIBER   ---> kz          ~ -----> z
#          ~                            ~
#          ##############################
#        
#        
#         let particles has the same hight -> rho = rho'


# ## Creating variables
P_laser = 100e-3  # [W]
R_focus = 1e-6  # [m]

Intensity = P_laser / (np.pi * R_focus**2)  # [W/m^2]

E0_charastic = np.sqrt(0.5 * const.Z0 * Intensity)
print('E_0 = %.1e' % (E0_charastic))

# theta = 30 * np.pi / 180
#E0_charastic = 1e4  # [V/m]
#time_charastic = 10e-3  # [s]
a_charastic = 0.1e-6 # [m]
# Silicon
# density_of_particle = 2328.  # [kg / m^3]
# wave_length = 350e-9  # [nm]
# wave_length_wg = 450e-9  # [nm]
gamma = 0.  # phenomenological parameter
epsilon_fiber = 2.09  # SiO2
epsilon_m = 1.  # Air


# fiber radius
# Wu and Tong, Nanophotonics 2, 407 (2013)
rho_c = 0.2e-6  # [m]

# must be int
n_mode = 1  # for HE11

gap = 5e-9  # [m]


def fd_dist(x, mu, T):
    return(1/(1+np.exp((x-mu)/T)))


def epsilon_particle(wl):
    # Polystyrene
    return(2.5)

def alpha0(wl):
    return(4 * np.pi * const.epsilon0 * a_charastic**3 * 
           (epsilon_particle(wl) - epsilon_m) / (epsilon_particle(wl) + 2 * epsilon_m))

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
    return(np.sqrt(epsilon_fiber + 0j) * k1_wg(wl))

omega_c = 2*np.pi / rho_c * 2 / (np.sqrt(epsilon_fiber + 0j) + 1) * const.c
def kz_wg(omega):
    koef = (1 - fd_dist(omega, omega_c, omega_c*0.1)) * (np.sqrt(epsilon_fiber + 0j) - 1) + 1
    return(koef / const.c * omega)

def krho1_wg(wl):
    return(np.sqrt(k1_wg(wl)**2 - kz_wg(2*np.pi*const.c/wl)**2  + 0j))
def krho2_wg(wl):
    return(np.sqrt(k2_wg(wl)**2 - kz_wg(2*np.pi*const.c/wl)**2  + 0j))

def A_(wl):
    return(1/krho2_wg(wl)**2 - 1/krho1_wg(wl)**2)
def B_(wl):
    return(sp.jvp(n_mode, krho2_wg(wl) * rho_c) / 
           (krho2_wg(wl) * sp.jv(n_mode, krho2_wg(wl) * rho_c)))
def C_(wl):
    return(sp.h1vp(n_mode, krho1_wg(wl) * rho_c) / 
           krho1_wg(wl) * sp.hankel1(n_mode, krho1_wg(wl) * rho_c))
def A_prime(wl):
    kzwg = kz_wg(2*np.pi*const.c/wl)
    return(2 * (kzwg / krho2_wg(wl)**4 - kzwg / krho1_wg(wl)**4))
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
#    print('A = ', np.linalg.norm(A))
#    print('B = ', np.linalg.norm(B))
#    print('C = ', np.linalg.norm(C))
#    print('Ap = ', np.linalg.norm(Ap))
#    print('Bp = ', np.linalg.norm(Bp))
#    print('C = ', np.linalg.norm(Cp))
    
    
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
    kzwg = kz_wg(2*np.pi*const.c/wl)
    
    P = (kzwg/kr2**2 - kzwg/kr1**2)**2 * epsilon_fiber - \
        (Jnp2 / (kr2 * Jn2) - Hnp1 / (kr1 * Hn1)) *\
        (Jnp2 * k2**2 / (Jn2 * kr2) - Jnp1 * k1**2 / (Jn1 * kr1)) * rho_c**2
        
    #print('kr1rc=', kr1rc)
    #print('kr2rc=', kr2rc)
    #print('F * K^3 = ', F * KONSTANTA**3)
    #print('P / K = ', Jn1 / (F * Hn1) * P / KONSTANTA)
    
    return(Jn1 / (F * Hn1) * P)

def Gszz(rho, rho_prime, dz, wl):
    HH = sp.hankel1(n_mode, krho1_wg(wl)*rho) * \
         sp.hankel1(n_mode, krho1_wg(wl)*rho_prime)
    G = -0.5 * P11NN(wl) * krho1_wg(wl)**2 / (k1_wg(wl)**2) * \
        HH * np.exp(1j * kz_wg(2*np.pi*const.c/wl) * np.abs(dz))
    return(G)

######### FROM MATLAB CODE ##########
######### STATUS: CHECKED! ##########
def GwgazzF(k, eps1, eps2, rhoc, hight_under_fiber, delta_z, betaval, n=1):
    # function y = GwgazzF(k, eps1, eps2, rhoc, r1_vec, r2_vec, n, betaval)
    # - 04.07.2017 -
    # Calculates waveguided mode contribution to the GF for a given mode 'n'
    # k - k-vector value;
    # eps1 - outside; eps2 - inside; rhoc - radius;
    # r1_vec - reciever; r2_vec - source;
    # n - order of the mode
    # i, j - tensor indicies over cylindrical coordinates \rho, \phi, z;
    # tol - relative tolerance for the sum; |G^{N} - G^{N-1}|/G^{N}; 
    # betaval - value of the propagation constant

    r1 = rhoc + hight_under_fiber
    r2 = rhoc + hight_under_fiber
    p1 = 0
    p2 = 0
    z1 = 0
    z2 = delta_z

    a = np.sqrt(eps1*k**2 - betaval**2 + 0j)
    b = np.sqrt(eps2*k**2 - betaval**2 + 0j)

    Hn1r = sp.hankel1(n, a*r1)
    Hn1s = sp.hankel1(n, a*r2)

    DJna = sp.jvp(n, a*rhoc)
    Jna = sp.jv(n, a*rhoc)

    k1 = np.sqrt(eps1 + 0j)*k
    k2 = np.sqrt(eps2 + 0j)*k
    
    DJnb = sp.jvp(n, b*rhoc)
    Jnb = sp.jv(n, b*rhoc)
    DDJnb = sp.jvp(n, b*rhoc, 2)
    
    DHna = sp.h1vp(n, a*rhoc)
    Hna = sp.hankel1(n, a*rhoc)
    DDHna = sp.h1vp(n, a*rhoc, 2)
   
    A = b**(-2) - a**(-2)
    B = DJnb/(b*Jnb)
    C = DHna/(a*Hna)
    
    AP = 2*betaval*( b**(-4) - a**(-4) )
    BP = -(DDJnb*betaval*rhoc)/(b**2 * Jnb) + \
        ( DJnb*betaval*(Jnb/b + rhoc*DJnb) )/(b**2 * Jnb**2)
    CP = -(DDHna*betaval*rhoc)/(a**2 * Hna) + \
        ( DHna*betaval*(Hna/a + rhoc*DHna) )/(a**2 * Hna**2)
    
    DDT = -2*A*AP*betaval**2 * n**2 - 2*A**2*betaval*n**2 + \
        ((BP - CP)*(k2**2 * B - k1**2 * C) +
         (B - C)*(k2**2 * BP - k1**2 * CP) ) * rhoc**2
 
    # Fresnel coefficients 
    Rnum11nn = (Jna/Hna)*( n**2 * betaval**2 * ( b**(-2) - a**(-2) )**2 - \
               (DJnb/(b*Jnb) - DHna/(a*Hna) ) * \
               ( k2**2 * DJnb/(b*Jnb) - k1**2 * DJna/(a*Jna) )*rhoc**2)
    y = -1/2.0*Rnum11nn/DDT * \
        a**2 / k1**2 * Hn1r * Hn1s * np.cos(n*(p1 - p2))*np.exp(1j*betaval*np.abs(z1 - z2))
    if n == 0:
        y /= 2
    
    return(y)


# mu_12 = alpha_eff * E0
# ( 1,2 -- only if rho1 = rho2 )
def alpha_eff(rho1, rho2, z, wl):
    # Gs(r1, r2) = Gs(r2,r1)
    GGG = Gszz(rho1, rho2, 0, wl) + Gszz(rho1, rho2, z, wl) + \
          G0zz(rho1, 0., 0., rho2, 0., z, wl)
    return(alpha0(wl) / (1 - alpha0(wl) * k1_inc(wl)**2 / const.epsilon0 * GGG))
    #return(alpha0(wl)**2 * k1_inc(wl)**2 / const.epsilon0 * GGG)

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

def force_grad_scat(rho1, rho2, z, wl):
    al_eff = alpha_eff(rho1, rho2, z, wl)
    al_eff_RE = al_eff.real
    al_eff_IM = al_eff.imag
    
    Fz_grad = 0.5 * al_eff_RE**2 * \
         E0_charastic * E0_charastic.conjugate() / const.epsilon0
         
    Fz_scat = 0.5 * al_eff_IM**2 * \
         E0_charastic * E0_charastic.conjugate() / const.epsilon0
         
    z_plus = z + dz
    z_minus = z - dz
    
    G_plus =  Gszz(rho1, rho2, z_plus, wl) + \
              G0zz(rho1, 0., 0., rho2, 0., z_plus, wl)
              
    G_minus = Gszz(rho1, rho2, z_minus, wl) + \
              G0zz(rho1, 0., 0., rho2, 0., z_minus, wl)
              
    RE = k1_inc(wl)**2 * (G_plus - G_minus) * 0.5 / dz
    
    return(Fz_grad.real * RE.real, Fz_scat.real * RE.real)


def plot_F_wl(rho1, rho2, wl):
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7.24, 4.24))
    
    z_near = np.linspace(2.5*a_charastic, 7*a_charastic, 10)
    f = np.zeros([len(z_near), len(wl)])
    cmap = mpl.cm.autumn
    for i, z in enumerate(z_near):
        f[i] = force(rho1, rho2, z, wl)
        if i == 0 or i == len(z_near)-1:
            plt.plot(wl*1e9, f[i], label='dz/a = %.1f'%(z/a_charastic), color=cmap(i / float(len(z_near))))
        else:
            plt.plot(wl*1e9, f[i], color=cmap(i / float(len(z_near))))

    #plt.ylim(-1e-20, 1e-20)
    plt.title(r'a = %.0f nm, $\rho_c$ = %.0f nm, $\Delta z$ = %.1f $\mu$m' % (a_charastic*1e9, rho_c*1e9, z*1e6), loc='right')
    plt.xlabel(r'$\lambda$, nm')
    plt.ylabel(r'$Fz$, N')
    plt.legend()
    plt.grid()
    plt.show()

def plot_F(rho1, rho2, z, wl):
    f = force(rho1, rho2, z, wl)
    f_grad, f_scat = force_grad_scat(rho1, rho2, z, wl)
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7.24, 4.24))
    plt.plot(z/a_charastic, f, label='total')
    plt.plot(z/a_charastic, f_grad, label='gradient part', linestyle='--', alpha=.5)
    plt.plot(z/a_charastic, f_scat, label='scatering part', linestyle='--', alpha=.5)
    plt.title(r'a = %.0f nm, $\rho_c$ = %.0f nm, $\lambda$ = %.1f nm' % (a_charastic*1e9, rho_c*1e9, wl*1e9), loc='right')
    plt.xlabel(r'$\Delta z /a$')
    plt.ylabel(r'$Fz$, N')
    plt.legend()
    #plt.ylim(-1e-18, 1e-18)
    plt.grid()
    plt.show()
    
def plot_alpha_z(rho1, rho2, z, wl):
    al = alpha_eff(rho1,rho2, z, wl)/alpha0(wl)
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7.24, 4.24))
    plt.plot(z/a_charastic, al.real, label='Re')
    plt.plot(z/a_charastic, al.imag, label='Im', linestyle='--')
    plt.legend()
    plt.title(r'a = %.0f nm, $\rho_c$ = %.0f nm, $\lambda$ = %.1f nm' % (a_charastic*1e9, rho_c*1e9, wl*1e9), loc='right')
    plt.xlabel(r'$\Delta z/a$')
    plt.ylabel(r'$\alpha_{eff} / \alpha_0$')
    #plt.ylim(-1e-18, 1e-18)
    plt.grid()
    plt.show()    

def plot_G_z(rho1, rho2, z, wl):
    #al = alpha_eff(rho1,rho2, z, wl)
    k = 2 * np.pi / wl
    al = G0zz(rho1, 0., 0., rho2, 0., z, wl) / k
    al2 = Gszz(rho1, rho2, z, wl) / k
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7.24, 4.24))
#    plt.plot(z/a_charastic, al.real, label=r'Re $G_0$', color='red')
#    plt.plot(z/a_charastic, al.imag, label=r'Im $G_0$', color='red', linestyle='--', alpha=0.4)
#    plt.plot(z/a_charastic, al2.real, label=r'Re $G_s$', color='blue')
#    plt.plot(z/a_charastic, al2.imag, label=r'Im $G_s$', color='blue', linestyle='--', alpha=0.4)
    
    plt.plot(z/wl, al.real, label=r'Re $G_0$', color='red')
    plt.plot(z/wl, al.imag, label=r'Im $G_0$', color='red', linestyle='--', alpha=0.4)
    plt.plot(z/wl, al2.real, label=r'Re $G_s$', color='blue')
    plt.plot(z/wl, al2.imag, label=r'Im $G_s$', color='blue', linestyle='--', alpha=0.4)
    
    plt.legend()
    plt.title(r'a = %.0f nm, $\rho_c$ = %.0f nm, $\lambda$ = %.1f nm' % (a_charastic*1e9, rho_c*1e9, wl*1e9), loc='right')
    #plt.xlabel(r'$\Delta z/a$')
    plt.xlabel(r'$\Delta z/\lambda$')
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
    
# single mode criteria
def VVV_q(wl):
    V = 2*np.pi/wl * rho_c * np.sqrt(epsilon_fiber - epsilon_m)
    Vcr = 2.405
    lam_c = 1/Vcr * 2*np.pi * rho_c * np.sqrt(epsilon_fiber - epsilon_m)
    if V < Vcr:
        print('Single mode condition: PASSED!')
        #print('V/Vc = %.3f/2.405 < 1'% V)
    else:
        print('Single mode condition: FAILED!')
        #print('V/Vc = %.3f/2.405 > 1'% V)
    print('lambda critical = %.1f' % (lam_c * 1e9))

#plt.xkcd()        # on
#plt.rcdefaults()  # off
wl = 600e-9
VVV_q(wl)
hight = rho_c + gap + a_charastic

plot_F_wl(hight, hight, np.linspace(400, 1300, 400)*1e-9)
plot_F(hight, hight, np.linspace(1.5*a_charastic, 14*a_charastic, 300), wl)
plot_alpha_z(hight, hight, np.linspace(1.5*a_charastic, 14*a_charastic, 300), wl)
plot_G_z(hight, hight, np.linspace(4*a_charastic, 50*a_charastic, 300), wl)

acrticle_data_gamma = np.array([
        [0, 1.1999999999999997],
        [0.05962059620596194, 1.204081632653061],
        [0.10840108401084003, 1.1959183673469385],
        [0.14634146341463405, 1.1714285714285713],
        [0.17344173441734423, 1.1346938775510202],
        [0.19512195121951215, 1.093877551020408],
        [0.21138211382113825, 1.0448979591836733],
        [0.2384823848238482, 1.0122448979591834],
        [0.2872628726287263, 0.9918367346938772],
        [0.34688346883468824, 0.9959183673469385],
        [0.41192411924119243, 1.004081632653061],
        [0.4715447154471544, 1.008163265306122],
        [0.5474254742547424, 0.9918367346938772],
        [0.6124661246612464, 0.9755102040816326],
        [0.6991869918699187, 0.9632653061224488],
        [0.7750677506775068, 0.9755102040816326],
        [0.8455284552845528, 0.9877551020408162],
        [0.9159891598915988, 1.008163265306122],
        [0.9918699186991868, 1.004081632653061],
        [1.067750677506775, 0.9918367346938772],
        [1.149051490514905, 0.9918367346938772],
        [1.2411924119241193, 0.9918367346938772],
        [1.3333333333333333, 1.004081632653061],
        [1.4146341463414631, 1.0122448979591834],
        [1.4905149051490512, 1.0122448979591834],
        [1.5772357723577233, 0.9999999999999998],
        [1.6476964769647695, 0.9918367346938772],
        [1.7289972899728998, 0.9999999999999998],
        [1.8048780487804874, 1.004081632653061],
        [1.8753387533875336, 1.0122448979591834],
        [1.9566395663956635, 1.0163265306122446]])

def plot_Gzz_rho(rho_space, wl):
    G = Gszz(rho_space, rho_space, 0, wl)
    gamma = G.imag * 3 * wl 
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7.24, 4.24))
    #plt.plot(rho_space/wl, al.real + 1, label=r'Re $G$', color='red')
    plt.plot(rho_space/wl, gamma, label=r'$\Gamma$', color='red')
    #plt.plot(acrticle_data_gamma[:,0], acrticle_data_gamma[:,1], label='article data', linestyle='--')
    plt.legend()
    plt.title(r'$\rho_c$ = %.0f nm, $\lambda$ = %.1f nm' % (rho_c*1e9, wl*1e9), loc='right')
    plt.xlabel(r'$\Delta z/\lambda$')
    plt.ylabel(r'$F_p$')
    #plt.ylim(-1e-18, 1e-18)
    plt.grid()
    plt.show()

wl = 5 * rho_c
r0 = np.linspace(0.2*wl, 2*wl, 100)
plot_Gzz_rho(r0, wl)

def test():
    wl = 600e-9
    k = 2 * np.pi / wl
    G = Gszz(hight, hight, 5*wl, wl) / k
    print('|G| = ', np.linalg.norm(G))
