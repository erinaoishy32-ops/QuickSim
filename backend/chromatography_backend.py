import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import FVMtools as fvm  # Import custom FVM toolbox
import time
import argparse
import sys

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Chromatography simulator')
parser.add_argument('--L', type=float, default=10.0, help='Column length (cm)')
parser.add_argument('--eps_c', type=float, default=0.45, help='Column void fraction')
parser.add_argument('--eps_p', type=float, default=0.57, help='Packing porosity')
parser.add_argument('--Pe', type=float, default=0.5, help='Peclet number')
parser.add_argument('--v', type=float, default=300, help='Superficial velocity (cm/min)')
parser.add_argument('--N', type=int, default=120, help='Number of grid points')
parser.add_argument('--F', type=float, default=30, help='CV/h')
parser.add_argument('--dp', type=float, default=0.003, help='Particle diameter')
parser.add_argument('--t_load', type=float, default=0.5, help='Sample loading time (cv)')
parser.add_argument('--t_max', type=float, default=56.0, help='Total simulation time (CV)')
parser.add_argument('--c_feed_A', type=float, default=1.15, help='Feed concentration A (g/L)')
parser.add_argument('--c_feed_B', type=float, default=2.0, help='Feed concentration B (g/L)')
parser.add_argument('--c_feed_C', type=float, default=0.23, help='Feed concentration C (g/L)')
parser.add_argument('--c_feed_D', type=float, default=0.40, help='Feed concentration D (g/L)')
parser.add_argument('--H0_A', type=float, default=0.6e-4, help='H0 for component A')
parser.add_argument('--H0_B', type=float, default=8.5e-4, help='H0 for component B')
parser.add_argument('--H0_C', type=float, default=2.97e-4, help='H0 for component C')
parser.add_argument('--H0_D', type=float, default=9e-4, help='H0 for component D')
parser.add_argument('--beta_A', type=float, default=4.37, help='beta for component A')
parser.add_argument('--beta_B', type=float, default=5.17, help='beta for component B')
parser.add_argument('--beta_C', type=float, default=5.14, help='beta for component C')
parser.add_argument('--beta_D', type=float, default=5.93, help='beta for component D')
parser.add_argument('--q_max_A', type=float, default=70.0, help='q_max for component A')
parser.add_argument('--q_max_B', type=float, default=70.0, help='q_max for component B')
parser.add_argument('--q_max_C', type=float, default=70.0, help='q_max for component C')
parser.add_argument('--q_max_D', type=float, default=70.0, help='q_max for component D')
parser.add_argument('--k_kin_A', type=float, default=300.0, help='k_kin for component A')
parser.add_argument('--k_kin_B', type=float, default=75.0, help='k_kin for component B')
parser.add_argument('--k_kin_C', type=float, default=90.0, help='k_kin for component C')
parser.add_argument('--k_kin_D', type=float, default=42.0, help='k_kin for component D')

args = parser.parse_args()

# Parameters
L = args.L          # Column length (cm)
eps_c = args.eps_c      # Column void fraction [cite: 22]
eps_p = args.eps_p      # Packing porosity [cite: 22]
Pe = args.Pe          # Peclet number [cite: 22]
v = args.v         # Superficial velocity (cm/min)
N = args.N            # Number of grid points
dz = L / N        # Discretization step size (cm)
F = args.F            #CV/h
dp = args.dp

# Experimental conditions
t_load = args.t_load    # Sample loading time (cv)
t_max = args.t_max     # Total simulation time (CV)

# Calculate axial dispersion coefficient D_ax from Peclet number (cm²/min)
D_ax = (v * dp) / Pe


#Aproximation and boundary condition matrix
[A1, A1f] = fvm.FVMdisc1st(N, dz, '2pb')    # 2-point boundary scheme (convection)
[A2, A2f] = fvm.FVMdisc2nd(N, dz, '3pc')    # 3-point central scheme (diffusion)
[B1, B0] = fvm.FVMdiscBV(N, dz, [0, 1], [[1, 0], [0, 0]])  # Boundary condition matrix


#Component Data
comp_names = ['A', 'B', 'C', 'D']
H0 = np.array([args.H0_A, args.H0_B, args.H0_C, args.H0_D])  # Initial Henry's constant
beta = np.array([args.beta_A, args.beta_B, args.beta_C, args.beta_D])       # Salt gradient coefficient
q_max = np.array([args.q_max_A, args.q_max_B, args.q_max_C, args.q_max_D])      # Maximum adsorption capacity (g/L)
k_kin = np.array([args.k_kin_A, args.k_kin_B, args.k_kin_C, args.k_kin_D])     # Kinetic rate constant (1/h)

# --- 3. Experimental Conditions ---

#c_feed = np.array([3.84, 3.84, 3.84, 3.84])  # Feed concentration (g/L)
c_feed = np.array([args.c_feed_A, args.c_feed_B, args.c_feed_C, args.c_feed_D])  # Feed concentration (g/L)

#salt concentration determination
def get_conditions(tcv):
    """Define sample injection logic and salt gradient (gradient elution)"""
    if tcv < 0:
        # Injection phase: constant salt concentration 0.1M
        salt = 0.1
        cin = c_feed
    elif tcv<1:
        salt = 0.05
        cin = np.zeros(4)
    elif tcv<51:
        # Gradient elution: linear increase in salt concentration (capped at 0.4M to avoid excess)
        salt = 0.1 + 0.005 * (tcv - 1)
        cin = np.zeros(4)
        #salt = min(salt, 0.4)  # Limit maximum salt concentration
    else: 
        salt = 0.5
        cin = np.zeros(4)
    return cin, salt



# --- 4. ODE System (core chromatography equations) ---
def chromatography_ode(t, y):
    # y dimension: 4*N (mobile phase concentration C) + 4*N (stationary phase concentration q) = 8*N
    C = y[:4*N].reshape((4, N))  # Mobile phase concentration: (component, grid)
    q = y[4*N:].reshape((4, N))  # Stationary phase concentration: (component, grid)
    
    # Get feed concentration and salt concentration at current time
    c_in, salt = get_conditions(t*F)
    
    # Calculate salt-dependent Henry's constant H = H0 * salt^(-beta)
    H = H0[:, np.newaxis] * (salt ** (-beta[:, np.newaxis]))  # (4, N)
    '''plt.plot(t*F,salt,'o')'''
    
    # Competitive adsorption term: sum(q_j / q_max_j) (total adsorption ratio per grid)
    sum_q_ratio = np.sum(q / q_max[:, np.newaxis], axis=0)  # (N,)
    
    # Initialize concentration change rates
    dCdt = np.zeros((4, N))
    dqdt = np.zeros((4, N))
    
    # Phase volume ratio: stationary phase / mobile phase (core adsorption term coefficient)
    phase_ratio = ((1 - eps_c) / eps_c) * eps_p  

    # Calculate rate equations for each component
    for i in range(4):
        
        # 1. Adsorption kinetic rate (Langmuir competitive adsorption)
        dqdt[i, :] = k_kin[i] * (H[i, :] * C[i, :] * (1 - sum_q_ratio) - q[i, :])
        
        # 2. Convection-diffusion transport term (spatial discretization)
        transport = D_ax * (A2 @ C[i, :]) - (v * A1 @ C[i, :])
        # 3. Inlet boundary condition: supplement feed concentration (FVM boundary treatment)
        transport[0] += (v / dz) * c_in[i]  # Inlet mass flux
        
        # 4. Total mass conservation: mobile phase concentration change = transport term - adsorption term
        dCdt[i, :] = transport - phase_ratio * dqdt[i, :]
    
    # Flatten array for return (solve_ivp requires 1D output)
    return np.concatenate([dCdt.flatten(), dqdt.flatten()])




# --- 5. Numerical Solution ---
# Initial conditions: all concentrations are zero
y0 = np.zeros(8 * N)  

#print("Solving chromatography ODE system... (approx. 10 seconds, depending on computer performance)")
# Use LSODA solver (suitable for stiff ODEs, typical choice for chromatography equations)

start_time = time.time()
sol = solve_ivp(
    chromatography_ode, 
    [-t_load/F, t_max/F], 
    y0, 
    method='LSODA',
    t_eval=np.linspace(0, t_max/F, 300),  # Output 300 time points
    rtol=1e-5,  # Relative tolerance
    atol=1e-8,  # Absolute tolerance
    max_step=0.05  # Limit maximum time step to improve stability
)
stop_time = time.time()

# --- 6. Result Post-processing and Visualization ---
if sol.success:
    #print("Solution successful! Generating plot...")
    
    t = sol.t * F 
    outlet_conc = np.zeros((4, len(t)))
    for i in range(4):
        outlet_conc[i, :] = sol.y[i*N + (N-1), :]

    fig, ax1 = plt.subplots(figsize=(15, 10))

    # --- 绘制组分浓度 (左侧 Y 轴) ---
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(4):
        ax1.plot(t, outlet_conc[i, :], label=f'Component {comp_names[i]}', 
                 color=colors[i], linewidth=2.5)
    
    ax1.set_xlabel("Time (CV)", fontsize=16)
    ax1.set_ylabel("Outlet Concentration (g/L)", fontsize=16)
    ax1.legend(loc='lower right', fontsize=16)

   
    ax2 = ax1.twinx()  
    salt_profile = [get_conditions(time)[1] for time in t]
    ax2.plot(t, salt_profile, '--', color='gray', alpha=0.6, label='Salt Gradient')
    ax2.set_ylabel("Salt Concentration (M)", color='gray', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='gray')

    #plt.title("", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.savefig("backend/plots/chrom_fig1.png",bbox_inches='tight')

    #Output key results
#print("\n=== Simulation Results Summary ===")
for i in range(4):
    max_conc = np.max(outlet_conc[i, :])
    elution_time = t[np.argmax(outlet_conc[i, :])]
    print(f"Component {comp_names[i]}: Maximum concentration = {max_conc:.4f} g/L, Elution time = {elution_time:.2f} CV")
else:
    print("")
    
#print("Simulation starting time: ", start_time)
#print("Simulation ending time: ", stop_time)
print("Time needed for simulation: ", float(stop_time - start_time),"s")