# QuickSim
QuickSim is a Python-based Graphical User Interface (GUI) simulator designed to model and optimize bioprocessing and chemical systems. By integrating complex mathematical models with an intuitive frontend built using the Tkinter library, the platform allows users to test operating conditions and visualize results without requiring advanced programming skills. This computational approach reduces the time and high costs traditionally associated with large-scale physical lab experiments.

Distillation: Utilizes a robust tray model under non-isothermal and isobaric conditions to calculate liquid composition and temperature dynamics, governed by differential and algebraic equations.

Chromatography: Employs a one-dimensional packed-bed dispersion model. It accounts for axial concentration variations in the mobile phase via convection and dispersion, alongside Langmuir adsorption for the stationary phase.

Packed-Bed Reactors: Models a first-order irreversible exothermic reaction using both axial and radial dispersion. It incorporates heat generation, cooling terms, and temperature dependencies based on the Arrhenius equation.

Architecture and Methodology Numerical Methods: The core mathematics rely on the Finite Volume Method (FVM) for discretization and the Method of Lines (MOL) to simplify continuous partial differential equations (PDEs) into solvable ordinary differential equations (ODEs).

Interface Design: The GUI features a centralized starting window that branches into dedicated screens for each of the three simulators. Each screen provides structured input fields on the left and a scrollable output canvas on the right for generating and visualizing plotted results.

Limitations and Future Development While QuickSim successfully lowers the barrier to entry for process simulation, it currently operates with static backend models that restrict extensive structural modifications beyond parameter adjustments. Furthermore, the individual process modules are entirely independent, requiring manual data entry to move from one process to the next. Future iterations aim to integrate continuous multi-module simulations with automatic data transfer, add data storage functions for comparative analysis, and implement security features like user authentication to protect proprietary simulation data.
