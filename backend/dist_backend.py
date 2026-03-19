import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, fsolve
from scipy.integrate import solve_ivp
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Distillation simulator')
parser.add_argument('--alpha', type=float, default=2.5, help='Equilibrium constant')
parser.add_argument('--F', type=float, default=2268, help='Feed flow rate (kmole/h)')
parser.add_argument('--xfeed', type=float, default=0.144, help='Feed mole fraction')
parser.add_argument('--RD', type=float, default=1.66, help='Reflux ratio')
parser.add_argument('--V', type=float, default=1027, help='Vapor flow (kmole/h)')
parser.add_argument('--N', type=int, default=20, help='Number of trays')
parser.add_argument('--Nf', type=int, default=18, help='Feed tray location')
parser.add_argument('--M', type=float, default=20, help='Tray hold-up (kmole)')
parser.add_argument('--Mc', type=float, default=500, help='Condenser hold-up (kmole)')
parser.add_argument('--Mb', type=float, default=200, help='Boiler hold-up (kmole)')
parser.add_argument('--t_max', type=float, default=10, help='Simulation time (hours)')

args = parser.parse_args() 

def equilRelVol(x,alpha): 
    y = alpha*x/(1+(alpha-1)*x) 
    return y 
 
def colmodel(t,z,dd): 
 # compositions 
     x = z[:N] 
     xb = z[N] 
     x0 = z[N+1] 
 
 # equil calc 
     y = equilRelVol(x,dd['alpha']) 
     yb = equilRelVol(xb,dd['alpha']) 
 
 # Liquid composition from tray above 
     xim1 = np.hstack(( x0, x[:N-1] )) 
 
 # Vapour composition from tray below 
     yip1 = np.hstack(( y[1:], yb )) 
 
 # Liquid flows 
     LF = dd['L'] + \
         np.hstack(( np.zeros(dd['Nf']), 
                    dd['F']*np.ones(N-dd['Nf']) )) 
     FF = np.hstack(( np.zeros(dd['Nf']-1), 
                    1, 
                    np.zeros(N-dd['Nf']) )) *dd['F'] 
 
 # tray composition balances 
     dxdt = 1/dd['M']*(LF*(xim1-x) + dd['V']*(yip1-y) + FF*(dd['xfeed']-x)) 
 
 # boiler and drum composition balances 
     dxbdt = 1/dd['Mb']*(LF[-1]*(x[-1]-xb) + dd['V']*(xb-yb)) 
     dx0dt = 1/dd['Mc']*(dd['V']*(y[0]-x0)) 
 
 # derivatives 
     dzdt = np.hstack((dxdt,dxbdt,dx0dt)) 
     return dzdt 
 
# PARAMETERS 
dd = {} 
 
# *** Equilibrium data 
dd['alpha'] = args.alpha
 
# *** Flows 
#Feed 
dd['F'] = args.F  # kmole/h
dd['xfeed'] = args.xfeed  # mole fraction
 
# Reflux ratio 
dd['RD'] = args.RD
dd['V'] = args.V
dd['D'] = dd['V']/(1+dd['RD'])
dd['L'] = dd['RD']*dd['D'] 
 
# *** Column 
# Tray numbering 
N = args.N
dd['N'] = N 
dd['Nf'] = args.Nf
dd['Nr'] = dd['Nf']-1 # rectifier 
dd['Ns'] = N-dd['Nf'] # stripper 
 
# Tray hold-up 
dd['M'] = args.M  # kmole 
dd['Mc'] = args.Mc
dd['Mb'] = args.Mb 
 
# Simulation 
xinit=dd['xfeed']*np.ones(N+2) 
 
ss_sol=fsolve(lambda x: colmodel(0,x,dd),xinit) 
 
Tspan=[0, args.t_max]
dyn_sol=solve_ivp(lambda t,x: colmodel(t,x,dd),Tspan,xinit, method='BDF', dense_output=True
) 
dd['ss'] = ss_sol 
dd['dyn'] = dyn_sol 
 


def destcol_binary_ideal_plot(dd): 
    ss = dd['ss'] 
    dyn = dd['dyn'] 
    N = dd['N'] 

    plt.figure(figsize=(6, 10))
    plt.subplots_adjust(
        left=0.1,    
        right=0.9,   
        bottom=0.1,  
        top=0.9,     
        wspace=0.3,  
        hspace=0.4   
        )
    plt.subplot(3,1,1)
    plt.plot(dyn.t,dyn.y[:N,:].T) 
    plt.plot([0,dyn.t[-1]], dd['xfeed']*np.ones(2), ':') 
    plt.title('Tray composition') 

    plt.subplot(3,1,2)
    plt.plot(dyn.t, dyn.y[N:N+2,:].T,'--') 
    plt.plot([0, dyn.t[-1]], dd['xfeed']*np.ones(2), ':') 
    plt.title('Top and bottom composition') 

    # plt.subplot(2,2,3)
    # xj = np.arange(0, 1, 0.05) # start,stop, step yj=equilRelVol(xj, dd['alpha']) 
    # yj = equilRelVol(xj, dd['alpha'])
    # xx = np.zeros(2*(N+1)+1) 
    # xx[::2] = np.hstack( (ss[N+1], ss[:N], ss[N]) )
    # xx[1::2] = np.hstack( (ss[:N], ss[N] ) )
    # ysol = equilRelVol(ss[:N+1],dd['alpha']) 
    # yy = np.zeros(2*(N+1)+1) 
    # yy[::2] = np.hstack( (ysol[:N], ysol[N], ss[N]) ) 
    # yy[1::2] = np.hstack( (ysol[:N], ysol[N]) ) 
    # plt.plot([0, 1],[0, 1],':k') 
    # plt.plot(xj, yj, xx, yy) 
    # plt.title('Equilibrium plot') 
 
    plt.subplot(3,1,3)
    plt.plot(np.arange(0,N+2), np.hstack((ss[N+1], ss[:N], ss[N])) ) 
    plt.plot(dd['Nf'], ss[dd['Nf']-1], 'o') 
    plt.title('Column profile') 
    plt.savefig('backend/plots/dist_fig1.png')
    
destcol_binary_ideal_plot(dd)
#plt.show()