import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import argparse
import sys
from distillation import antoineEthanolWater, wilsonActFact
from integrate_ketn01_2026 import solve_ivp_mass  # keep your import

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Distillation simulator (non-ideal, with VLE)')
parser.add_argument('--P', type=float, default=101300, help='Pressure (Pa)')
parser.add_argument('--F', type=float, default=2268.0, help='Feed flow rate (kmole/h)')
parser.add_argument('--xfeed', type=float, default=0.144, help='Feed composition')
parser.add_argument('--RD', type=float, default=1.66, help='Reflux ratio')
parser.add_argument('--V', type=float, default=1027.0, help='Vapor flow (kmole/h)')
parser.add_argument('--N', type=int, default=20, help='Number of trays')
parser.add_argument('--Nf', type=int, default=18, help='Feed tray location')
parser.add_argument('--M', type=float, default=20.0, help='Tray hold-up (kmole)')
parser.add_argument('--Mc', type=float, default=500.0, help='Condenser hold-up (kmole)')
parser.add_argument('--Mb', type=float, default=200.0, help='Boiler hold-up (kmole)')
parser.add_argument('--t_max', type=float, default=200.0, help='Simulation time (hours)')

args = parser.parse_args()

# --------------------------
# VLE: Antoine + Wilson
# --------------------------
def ActFact(x, T, P):
    p01, p02 = antoineEthanolWater(T)
    gamma1, gamma2 = wilsonActFact(x, T)
    y1 = gamma1 * p01 * x / P
    y2 = gamma2 * p02 * (1 - x) / P
    return y1, y2

def bubble_point_T(x, Tguess, P):
    # Solve 1 - y1 - y2 = 0 for T (scalar)
    def res(T):
        y1, y2 = ActFact(x, T, P)
        return 1.0 - y1 - y2
    return float(fsolve(res, Tguess)[0])

def vle_curve(xgrid, Tguess, P):
    ygrid = np.zeros_like(xgrid)
    Tgrid = np.zeros_like(xgrid)
    Tprev = Tguess
    for k, x in enumerate(xgrid):
        Tk = bubble_point_T(float(x), Tprev, P)
        y1, y2 = ActFact(float(x), Tk, P)
        ygrid[k] = float(y1)
        Tgrid[k] = Tk
        Tprev = Tk
    return ygrid, Tgrid

# --------------------------
# Column model (Tray Model II DAE)
# --------------------------
def colmodel(t, z, dd):
    N = dd["N"]
    Nf = dd["Nf"]

    # states
    x  = z[:N]
    xb = z[N]
    xD = z[N+1]
    T  = z[N+2:N+2+N]
    Tb = z[N+2+N]

    # equilibrium
    y, y2 = ActFact(x, T, dd["P"])
    yb, yb2 = ActFact(np.array([xb]), np.array([Tb]), dd["P"])
    yb = float(yb)     # IMPORTANT FIX
    yb2 = float(yb2)   # IMPORTANT FIX

    # reflux from RD and V (constant)
    L = dd["RD"] / (1.0 + dd["RD"]) * dd["V"]
    dd["L"] = L  # keep it in dict if you want

    # inlet compositions
    xim1 = np.hstack((xD, x[:N-1]))
    yip1 = np.hstack((y[1:], yb))

    # flows
    LF = L + np.hstack((np.zeros(Nf), dd["F"] * np.ones(N - Nf)))
    FF = np.zeros(N)
    FF[Nf-1] = dd["F"]

    # dynamics
    dxdt = (1/dd["M"])  * (LF*(xim1 - x) + dd["V"]*(yip1 - y) + FF*(dd["xfeed"] - x))
    dxbdt = (1/dd["Mb"]) * (LF[-1]*(x[-1] - xb) + dd["V"]*(xb - yb))
    dxDdt = (1/dd["Mc"]) * (dd["V"]*(y[0] - xD))

    # algebraic residuals: y1 + y2 = 1 (each stage)
    req  = 1.0 - (y + y2)          # length N
    reqb = np.array([1.0 - (yb + yb2)])  # length 1 (keep consistent)

    return np.hstack((dxdt, dxbdt, dxDdt, req, reqb))

def destcol_binary_nonideal():
    dd = {}
    dd["P"] = args.P
    dd["F"] = args.F
    dd["xfeed"] = args.xfeed
    dd["RD"] = args.RD
    dd["V"] = args.V

    # trays/holdups
    dd["N"] = args.N
    dd["Nf"] = args.Nf
    dd["M"]  = args.M
    dd["Mc"] = args.Mc
    dd["Mb"] = args.Mb

    N = dd["N"]

    # initial conditions (start-up)
    xinit = dd["xfeed"] * np.ones(N + 2)
    Tinit = (273.15 + 80.0) * np.ones(N + 1)
    init = np.hstack((xinit, Tinit))

    # DAE mass matrix
    n_ode = N + 2
    n_ae  = N + 1
    M11 = np.eye(n_ode)
    M12 = np.zeros((n_ode, n_ae))
    M2  = np.zeros((n_ae, n_ode + n_ae))
    MM  = np.vstack((np.hstack((M11, M12)), M2))

    # simulate start-up
    Tspan = [0.0, args.t_max]
    dyn = solve_ivp_mass(lambda t, x: colmodel(t, x, dd), Tspan, init, n=2000, mass=MM)
    dd["dyn"] = dyn

    # steady-state (optional)
    ss_sol = fsolve(lambda x: colmodel(0.0, x, dd), init)
    dd["ss"] = ss_sol
    return dd

def plot_results(dd):
    dyn = dd["dyn"]
    N = dd["N"]

    plt.figure(figsize=(8, 18))

    # Tray compositions
    plt.subplot(3, 1, 1)
    plt.plot(dyn.t, dyn.y[:N, :].T)
    plt.title("Tray Concentrations")
    plt.xlabel("Time")
    plt.ylabel("x (ethanol in liquid)")

    # Drum and reboiler
    plt.subplot(3, 1, 2)
    plt.plot(dyn.t, dyn.y[N:N+2, :].T, "--")
    plt.plot([dyn.t[0], dyn.t[-1]], dd["xfeed"] * np.ones(2), ":k")
    plt.title("Bottom (xB) and Top (xD)")
    plt.legend(["xB", "xD", "Feed"])


    # Final column profile
    plt.subplot(3, 1, 3)
    x_final = dyn.y[:N+2, -1]  # only x-states
    xD = x_final[N+1]
    xb = x_final[N]
    xt = x_final[:N]
    plt.plot(np.arange(0, N+2), np.hstack((xD, xt, xb)))
    plt.plot(dd["Nf"], xt[dd["Nf"]-1], "o")
    plt.title("Final Column Profile (xD, trays, xB)")
    plt.savefig('backend/plots/dist_fig1.png',bbox_inches='tight')

    plt.tight_layout()
    #plt.show()

if __name__ == "__main__":
    dd = destcol_binary_nonideal()
    plot_results(dd)
    
    # print("\n=== Distillation Simulation Results (Non-Ideal) ===")
    # print(f"Pressure: {dd['P']/1000:.1f} kPa")
    # print(f"Number of trays: {dd['N']}")
    # print(f"Feed tray location: {dd['Nf']}")
    # print(f"Feed composition: {dd['xfeed']:.4f}")
    # print(f"Reflux ratio: {dd['RD']:.2f}")
    # print(f"Simulation time: {args.t_max:.1f} hours")
    # print("Plot saved to: backend/plots/dist_fig1.png")
