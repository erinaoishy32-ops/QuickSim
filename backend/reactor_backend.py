import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse
import sys
import os

# Add the current directory to the path so we can import FVMtools2D_v4
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from FVMtools2D_v4 import FVM2D

plt.close('all')
def reactormodel(t, y, p, f):

    #adsorbtion_term new added !!!!!!!!!!!!!!!!!!!!!!!!!
    phase_ratio = ((1 - p.eps_c) / p.eps_p)
    a_term = phase_ratio * p.eff_fac

    cA = y[:p.Nz*p.Nr]
    T = y[p.Nz*p.Nr:]
    kr = p.A*np.exp(-p.E/(p.R*T))
    rA = - kr * cA
    Qr = (-p.deltaHr)*(-rA)
    
   
    r_vec = f.rprofile(p.r) 
   
    radial_diffusion_A = p.Dax * (f.d2_dr2(cA) + (1/r_vec) * f.d_dr(cA))
    dcAdt = p.Dax * f.d2_dz2(cA) + radial_diffusion_A - (p.u/p.eps_p) * f.d_dz(cA) + rA * a_term
   
    radial_conduction_T = (p.k / (p.rho * p.Cp)) * (g.d2_dr2(T) + (1/r_vec) * g.d_dr(T))
    axial_conduction_T = (p.k / (p.rho * p.Cp)) * g.d2_dz2(T)
    convection_T = - (p.u0 / p.eps_p) * g.d_dz(T)
    reaction_heat_T = Qr / (p.rho * p.Cp)
    
    dTdt = convection_T + axial_conduction_T + radial_conduction_T + reaction_heat_T
    
    dydt = np.hstack((dcAdt, dTdt))
    return dydt




class Parameters(): # Empty class to store parameters in instead of dict
    pass

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Packed-bed Reactor Simulator')
parser.add_argument('--L', type=float, default=2, help='Reactor Length (m)')
parser.add_argument('--Rr', type=float, default=0.5, help='Reactor Radius (m)')
parser.add_argument('--eps_c', type=float, default=0.33, help='Void fraction')
parser.add_argument('--poro', type=float, default=0.5, help='Porosity')
parser.add_argument('--cAin', type=float, default=25, help='Feed Concentration (mol/m3)')
parser.add_argument('--Tin', type=float, default=353, help='Feed temperature (K)')
parser.add_argument('--fflow', type=float, default=50, help='Feed flow rate (m3/h)')
parser.add_argument('--A', type=float, default=240000000, help='Frequency Factor (/s)')
parser.add_argument('--E', type=float, default=70000, help='Activation Energy (J/mol)')
parser.add_argument('--deltaHr', type=float, default=-180000, help='Reaction Heat (J/mol)')
parser.add_argument('--rho', type=float, default=4, help='Gas density (kg/m3)')

args = parser.parse_args()

# p is used for parameters
p = Parameters()
p.L = args.L
p.Rr = args.Rr

p.cAin = args.cAin

# Convert feed flow rate from m3/h to m3/s
p.F0 = args.fflow / 3600  # Convert m3/h to m3/s
p.u0 = p.F0 / (np.pi * p.Rr**2)  # [m/s]

p.p_rad = 1e-4  # [particle radius m]
p.poro = args.poro  # [porosity]
p.pecn = 0.5  # [peclet number]

p.eps_c = args.eps_c  # [void]
p.eps_p = p.poro
p.eff_fac = 1  # [effectiveness factor]

# Calculate axial dispersion coefficient D_ax from Peclet number
p.Dax = (p.u0 * p.p_rad) / p.pecn

p.R = 8.314  # [J/mol/K]
p.E = args.E  # [J/mol] Activation energy
p.A = args.A  # [/s] Frequency factor
p.Keq0 = 1000  # [-]

p.deltaHr = args.deltaHr  # [J/mol] Reaction heat

p.rho = args.rho  # [kg/m^3]
p.Cp = 1000 / 0.03  # [J/kg/K]
p.k = 1.0  # [W/m/K]

p.Tin = args.Tin  # Feed temperature [K]
p.Tc = 293  # wall temperature[K]

p.hht = 13  # [J/m^2/s/K]
p.Aht = 2 * np.pi * p.Rr * p.L  # [m2]
p.V = np.pi * p.Rr**2 * p.L  # [m3]
p.alpha_outer = 1300  # [J/m2/s/K]


p.Nz = 30
p.hz = p.L/p.Nz
p.Nr = 20
p.hr = p.Rr/p.Nr


# f is used for concentration transport
f = FVM2D(p.Nz, p.Nr, p.hz, p.hr, Atype1y='2pf', xname='z', yname='r')
f.set_bv_z0(0, 1, -1) # dirichlet at inlet
f.set_z0_values(p.cAin)

# g is used for heat transport
g = FVM2D(p.Nz, p.Nr, p.hz, p.hr, Atype1y='2pf', xname='z', yname='r')
g.set_bv_z0(0, 1, -1) # dirichlet at inlet
g.set_bv_rL(1, p.alpha_outer/p.k, -p.alpha_outer/p.k) # flux at wall
g.set_z0_values(p.Tin)
g.set_rL_values(p.Tc)

# Create the velocity profile in r direction
p.r = np.linspace(p.hr/2, p.Rr-p.hr/2, p.Nr)
p.z = np.linspace(p.hz/2, p.L-p.hz/2, p.Nz)
u = 2*p.u0*(1-(p.r/p.Rr)**2)

# Make a flattened vector (Nx*Ny) of the rprofile to be used in the eqs.
p.u = f.rprofile(u)



# Solve
tspan = [0, 100]
cAinit = p.cAin*np.ones((p.Nr, p.Nz)).flatten()
Tinit = p.Tin*np.ones((p.Nr, p.Nz)).flatten()
yinit = np.hstack((cAinit, Tinit))

y_typical = yinit
JP = g.FVMjpattern(lambda t, cA: reactormodel(t, cA, p, f), len(yinit),
y_typical=y_typical)

tic = time.time()
sol = solve_ivp(lambda t, cA: reactormodel(t, cA, p, f),\
tspan, yinit, method = 'BDF', jac_sparsity=JP)#, rtol=1e-6, atol=1e-9,)
toc = time.time() - tic
#print(toc)
t = sol.t
y = sol.y.T

cA = y[:,:p.Nr*p.Nz]
T = y[:,p.Nr*p.Nz:]

cAstat = cA[-1,:]
cAstat_m = f.reshape(cAstat)

Tstat = T[-1,:]
Tstat_m = g.reshape(Tstat)
cCsol = 2*(p.cAin-cAstat)




fig,ax0 = plt.subplots(2,1)
f.plotContour(cAstat, ax=ax0[0])
g.plotContour(Tstat, ax=ax0[1])
ax0[0].set_title('Concentration')
ax0[1].set_title('Temperature')
plt.savefig('backend/plots/reactor_fig1.png')

ax1 = f.plotAxisymmetric(cAstat)
ax1.set_title('Concentration')
plt.savefig('backend/plots/reactor_fig2.png')

ax2 = g.plotAxisymmetric(Tstat)
ax2.set_title('Temperature')
plt.savefig('backend/plots/reactor_fig3.png')

ax3 = g.plotContour(p.u)
ax3.set_title('Flow profile')
plt.savefig('backend/plots/reactor_fig4.png')

int_Tur = np.trapz(Tstat_m[:,-1]*u*p.r, g.rg[:,-1])
int_ur = np.trapz(u*p.r, g.rg[:,-1])
Tout = int_Tur/int_ur
print(f'Mean temperature out is {Tout:.1f} K')
int_cAur = np.trapz(cAstat_m[:,-1]*u*p.r, g.rg[:,-1])
int_ur = np.trapz(u*p.r, g.rg[:,-1])
cAout = int_cAur/int_ur
print(f'Mean concentration A out is {cAout:.1f} mol/m^3')

fig,ax4 = plt.subplots(2,1)
ax4[0].plot(g.rg[:,-1],Tstat_m[:,-1], label='Temp')
ax4[0].plot(g.rg[:,-1],Tout*np.ones(p.Nr), label='Mean temp')
ax4[0].set_xlabel('r [m]')
ax4[0].set_ylabel('T [K]')
ax4[0].set_title('Mean Temperature & Concentration profile')
ax4[1].plot(g.rg[:,-1],cAstat_m[:,-1], label='Conc A')
ax4[1].plot(g.rg[:,-1],cAout*np.ones(p.Nr), label='Mean conc A')
ax4[1].set_xlabel('r [m]')
ax4[1].set_ylabel('c [mol/m^3]')
plt.savefig('backend/plots/reactor_fig5.png')

plt.figure()
plt.plot(sol.t,sol.y.T)
plt.title('Dynamic plot for all states.')
#plt.show()
plt.savefig('backend/plots/reactor_fig6.png')

#(Conversion)
conversion = (1 - cAout / p.cAin) * 100
print(f"AVG conversion: {conversion:.2f} %")






