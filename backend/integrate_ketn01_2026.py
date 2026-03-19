import numpy as np
from scipy.optimize import root


class Sol:
    pass


class DAESolver:
    """
    Implicit Euler stepping for DAEs via a per-step nonlinear solve.

    Performance improvements:
      - In-place residual workspace (no split/concat allocations)
      - Optional reduced solve (y only) with dydt recovered from y
      - Tunable root() options and adaptive-step controller
    """
    def __init__(
        self,
        f,
        y0,
        dydt0,
        t0,
        tf,
        atol=1e-6,
        rtol=1e-3,
        max_step=0.1,
        min_step=1e-9,
        root_method='hybr',
        xtol=None,
        maxfev=None,
        safety=0.9,
        min_factor=0.2,
        max_factor=5.0,
        reduced=False,
    ):
        self.f = f
        self.y = np.array(y0, dtype=float)
        self.dydt = np.array(dydt0, dtype=float)
        self.t = float(t0)
        self.tf = float(tf)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.max_step = float(max_step)
        self.min_step = float(min_step)

        self.root_method = root_method
        self.xtol = xtol
        self.maxfev = maxfev
        self.safety = float(safety)
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)
        self.reduced = bool(reduced)

    def solve(self):
        tsol = []
        ysol = []

        atol = self.atol
        rtol = self.rtol
        t = self.t
        tf = self.tf
        y = self.y
        dydt = self.dydt

        h = self.max_step
        order = 1.0

        while t < tf:
            h = min(h, tf - t)

            ok, new_y, new_dydt, new_t = self.step(h, y, dydt, t)
            if not ok:
                h = max(self.min_step, h * 0.5)
                if h <= self.min_step:
                    raise RuntimeError("Step size too small.")
                continue

            diff = np.linalg.norm(new_y - y, ord=np.inf)
            scale = max(atol, np.linalg.norm(new_y, ord=np.inf) * rtol)
            err = diff / scale if scale > 0.0 else 0.0

            if err > 1.0:
                factor = max(self.min_factor, self.safety / err)
                h = max(self.min_step, h * factor)
                continue

            y = new_y
            dydt = new_dydt
            t = new_t
            tsol.append(t)
            ysol.append(y.copy())

            err_safe = max(err, 1e-12)
            grow = self.safety * err_safe ** (-1.0 / order)
            grow = min(self.max_factor, max(self.min_factor, grow))
            h = min(self.max_step, max(self.min_step, h * grow))

        self.y = y
        self.dydt = dydt
        self.t = t
        return tsol, ysol

    def step(self, h, y, dydt, t):
        """
        One implicit Euler step:
          y_{n+1} = y_n + h * dydt_{n+1}
          f(t_{n+1}, y_{n+1}, dydt_{n+1}) = 0
        """
        t_next = t + h
        f = self.f
        root_method = self.root_method
        options = {}
        if self.xtol is not None:
            options['xtol'] = self.xtol
        if self.maxfev is not None:
            options['maxfev'] = self.maxfev

        m = y.size
        y_guess = y + h * dydt

        if self.reduced:
            def residual_y(yn):
                return f(t_next, yn, (yn - y) / h)

            sol = root(residual_y, y_guess, method=root_method, options=options)
            if not sol.success:
                return False, None, None, None
            y_next = sol.x
            dydt_next = (y_next - y) / h
            return True, y_next, dydt_next, t_next

        z0 = np.empty(2 * m, dtype=float)
        z0[:m] = y_guess
        z0[m:] = dydt

        res = np.empty(2 * m, dtype=float)

        def residual_z(z):
            yn = z[:m]
            ydn = z[m:]
            res[:m] = yn - y - h * ydn
            res[m:] = f(t_next, yn, ydn)
            return res

        sol = root(residual_z, z0, method=root_method, options=options)
        if not sol.success:
            return False, None, None, None

        y_next = sol.x[:m].copy()
        dydt_next = sol.x[m:].copy()
        return True, y_next, dydt_next, t_next


class BDF2DAESolver:
    """
    BDF2 solver for DAEs defined by residual F(t, y, yd) = 0.
    - Variable-step BDF2 (second order) with correct variable-step coefficients.
    - Nonlinear solve only in y (m unknowns); yd is formed from BDF2 relation.
    - First step bootstrapped by an implicit Euler step.
    """
    def __init__(
        self,
        f,
        y0,
        dydt0,
        t0,
        tf,
        atol=1e-6,
        rtol=1e-3,
        max_step=0.1,
        min_step=1e-9,
        root_method='hybr',
        xtol=None,
        maxfev=None,
        safety=0.9,
        min_factor=0.2,
        max_factor=5.0,
    ):
        self.f = f
        self.y0 = np.array(y0, dtype=float)
        self.dydt0 = np.array(dydt0, dtype=float)
        self.t0 = float(t0)
        self.tf = float(tf)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.max_step = float(max_step)
        self.min_step = float(min_step)
        self.root_method = root_method
        self.xtol = xtol
        self.maxfev = maxfev
        self.safety = float(safety)
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)

    def _root_solve(self, fun, x0):
        opts = {}
        if self.xtol is not None:
            opts['xtol'] = self.xtol
        if self.maxfev is not None:
            opts['maxfev'] = self.maxfev
        sol = root(fun, x0, method=self.root_method, options=opts)
        return sol

    def _ie_step(self, t, y, h, dydt_guess):
        t1 = t + h
        y_guess = y + h * dydt_guess

        def res_ie(yn):
            yd = (yn - y) / h
            return self.f(t1, yn, yd)

        sol = self._root_solve(res_ie, y_guess)
        if not sol.success:
            return False, None, None, None
        y1 = sol.x
        yd1 = (y1 - y) / h
        return True, y1, yd1, t1

    @staticmethod
    def _bdf2_coeffs(h, h_prev):
        if h_prev is None or h_prev <= 0.0:
            r = 1.0
        else:
            r = h / h_prev
        a0 = (2.0 * r + 1.0) / (r + 1.0)
        a1 = -(r + 1.0)
        a2 = (r * r) / (r + 1.0)
        return a0, a1, a2

    def _bdf2_step(self, t, y_nm1, y_nm2, h, h_prev, dydt_nm1):
        t1 = t + h
        a0, a1, a2 = self._bdf2_coeffs(h, h_prev)

        def res_bdf2(yn):
            yd = (a0 * yn + a1 * y_nm1 + a2 * y_nm2) / h
            return self.f(t1, yn, yd)

        y_guess = y_nm1 + h * dydt_nm1
        sol = self._root_solve(res_bdf2, y_guess)
        if not sol.success:
            return False, None, None, None
        yn = sol.x
        yd = (a0 * yn + a1 * y_nm1 + a2 * y_nm2) / h
        return True, yn, yd, t1

    def solve(self):
        tsol = []
        ysol = []

        t = self.t0
        tf = self.tf
        y_nm2 = None
        y_nm1 = self.y0.copy()
        dydt_nm1 = self.dydt0.copy()

        h = self.max_step
        h_prev = None
        order_ie = 1.0
        order_bdf2 = 2.0

        while t < tf:
            h = min(h, tf - t)

            if y_nm2 is None:
                ok, y_new, dydt_new, t_new = self._ie_step(t, y_nm1, h, dydt_nm1)
                if not ok:
                    h = max(self.min_step, 0.5 * h)
                    if h <= self.min_step:
                        raise RuntimeError("IE bootstrap failed: step too small.")
                    continue

                diff = np.linalg.norm(y_new - y_nm1, ord=np.inf)
                scale = max(self.atol, np.linalg.norm(y_new, ord=np.inf) * self.rtol)
                err = diff / scale if scale > 0.0 else 0.0
                if err > 1.0:
                    factor = max(self.min_factor, self.safety / max(err, 1e-12))
                    h = max(self.min_step, h * factor)
                    continue

                y_nm2 = y_nm1
                y_nm1 = y_new
                dydt_nm1 = dydt_new
                t = t_new
                tsol.append(t)
                ysol.append(y_nm1.copy())

                grow = self.safety * max(err, 1e-12) ** (-1.0 / order_ie)
                grow = min(self.max_factor, max(self.min_factor, grow))
                h_prev = h
                h = min(self.max_step, max(self.min_step, h * grow))
                continue

            ok, y_new, dydt_new, t_new = self._bdf2_step(t, y_nm1, y_nm2, h, h_prev, dydt_nm1)
            if not ok:
                h = max(self.min_step, 0.5 * h)
                if h <= self.min_step:
                    raise RuntimeError("BDF2 solve failed: step too small.")
                continue

            diff = np.linalg.norm(y_new - y_nm1, ord=np.inf)
            scale = max(self.atol, np.linalg.norm(y_new, ord=np.inf) * self.rtol)
            err = diff / scale if scale > 0.0 else 0.0
            if err > 1.0:
                factor = max(self.min_factor, self.safety / max(err, 1e-12))
                h = max(self.min_step, h * factor)
                continue

            y_nm2 = y_nm1
            y_nm1 = y_new
            dydt_nm1 = dydt_new
            t = t_new
            tsol.append(t)
            ysol.append(y_nm1.copy())

            grow = self.safety * max(err, 1e-12) ** (-1.0 / order_bdf2)
            grow = min(self.max_factor, max(self.min_factor, grow))
            h_prev = h
            h = min(self.max_step, max(self.min_step, h * grow))

        return tsol, ysol


class SDIRK2DAESolver:
    """
    SDIRK2 (Alexander) solver for DAEs defined by residual F(t, y, yd) = 0.
    - 2-stage, L-stable, second-order SDIRK
    - Reduced stage solves: solve only for Y_i (m unknowns), compute Yd_i from SDIRK relation
    - Fallback: if step hits min_step, try a single implicit Euler step before failing
    """
    def __init__(
        self,
        f,
        y0,
        dydt0,
        t0,
        tf,
        atol=1e-6,
        rtol=1e-3,
        max_step=0.1,
        min_step=1e-9,
        root_method='hybr',
        xtol=None,
        maxfev=None,
        safety=0.9,
        min_factor=0.2,
        max_factor=5.0,
    ):
        self.f = f
        self.y = np.array(y0, dtype=float)
        self.dydt = np.array(dydt0, dtype=float)
        self.t = float(t0)
        self.tf = float(tf)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.max_step = float(max_step)
        self.min_step = float(min_step)
        self.root_method = root_method
        self.xtol = xtol
        self.maxfev = maxfev
        self.safety = float(safety)
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)

        self.gamma = 1.0 - 1.0 / np.sqrt(2.0)
        self.a11 = self.gamma
        self.a21 = 1.0 - self.gamma
        self.a22 = self.gamma
        self.b1 = 1.0 - self.gamma
        self.b2 = self.gamma
        self.c1 = self.gamma
        self.c2 = 1.0

        self.b1e = 1.0 - 2.0 * self.gamma
        self.b2e = 2.0 * self.gamma

    def _root(self, fun, z0):
        opts = {}
        if self.xtol is not None:
            opts['xtol'] = self.xtol
        if self.maxfev is not None:
            opts['maxfev'] = self.maxfev
        return root(fun, z0, method=self.root_method, options=opts)

    def _stage1(self, h, y, dydt, t):
        t1 = t + self.c1 * h
        y_guess = y + h * self.a11 * dydt

        def R1(Y1):
            Yd1 = (Y1 - y) / (h * self.a11)
            return self.f(t1, Y1, Yd1)

        sol = self._root(R1, y_guess)
        if not sol.success:
            return False, None, None
        Y1 = sol.x
        Yd1 = (Y1 - y) / (h * self.a11)
        return True, Y1, Yd1

    def _stage2(self, h, y, Yd1, t):
        t2 = t + self.c2 * h
        y_guess = y + h * (self.a21 * Yd1 + self.a22 * Yd1)

        def R2(Y2):
            Yd2 = (Y2 - y - h * self.a21 * Yd1) / (h * self.a22)
            return self.f(t2, Y2, Yd2)

        sol = self._root(R2, y_guess)
        if not sol.success:
            return False, None, None
        Y2 = sol.x
        Yd2 = (Y2 - y - h * self.a21 * Yd1) / (h * self.a22)
        return True, Y2, Yd2

    def _ie_fallback(self, h, y, dydt, t):
        t1 = t + h
        y_guess = y + h * dydt

        def Rie(Y1):
            Yd1 = (Y1 - y) / h
            return self.f(t1, Y1, Yd1)

        sol = self._root(Rie, y_guess)
        if not sol.success:
            return False, None, None, None
        y1 = sol.x
        yd1 = (y1 - y) / h
        return True, y1, yd1, t1

    def step(self, h, y, dydt, t):
        ok1, Y1, Yd1 = self._stage1(h, y, dydt, t)
        if not ok1:
            return False, None, None, None, None

        ok2, Y2, Yd2 = self._stage2(h, y, Yd1, t)
        if not ok2:
            return False, None, None, None, None

        y_next = y + h * (self.b1 * Yd1 + self.b2 * Yd2)
        yd_next = (self.b1 * Yd1 + self.b2 * Yd2)
        t_next = t + h

        y_emb = y + h * (self.b1e * Yd1 + self.b2e * Yd2)
        err_vec = y_next - y_emb
        return True, y_next, yd_next, t_next, err_vec

    def solve(self):
        tsol = []
        ysol = []

        t = self.t
        tf = self.tf
        y = self.y
        dydt = self.dydt

        h = self.max_step
        order = 2.0

        while t < tf:
            h = min(h, tf - t)

            ok, y_new, dydt_new, t_new, err_vec = self.step(h, y, dydt, t)
            if not ok:
                new_h = max(self.min_step, 0.5 * h)
                if new_h <= self.min_step + 0.0:
                    ok_ie, y_ie, yd_ie, t_ie = self._ie_fallback(h, y, dydt, t)
                    if not ok_ie:
                        raise RuntimeError("SDIRK step failed: step too small.")
                    y, dydt, t = y_ie, yd_ie, t_ie
                    tsol.append(t)
                    ysol.append(y.copy())
                    h = max(self.min_step, 0.5 * h)
                    continue
                h = new_h
                continue

            scale = np.maximum(self.atol, np.abs(y_new) * self.rtol)
            err = np.linalg.norm(err_vec / scale, ord=np.inf)

            if err > 1.0:
                factor = max(self.min_factor, self.safety * err ** (-1.0 / order))
                h = max(self.min_step, h * factor)
                continue

            y = y_new
            dydt = dydt_new
            t = t_new
            tsol.append(t)
            ysol.append(y.copy())

            err_safe = max(err, 1e-12)
            grow = self.safety * err_safe ** (-1.0 / order)
            grow = min(self.max_factor, max(self.min_factor, grow))
            h = min(self.max_step, max(self.min_step, h * grow))

        self.y = y
        self.dydt = dydt
        self.t = t
        return tsol, ysol


def implicit_euler(f, tspan, y0, yp0, dt, max_iter=50, tol=1e-6):
    """
    Solves a DAE system f(t, y, dydt) = 0 using the Implicit Euler method.

    Parameters:
        f (function): The DAE function f(t, y, dydt) = 0.
        tspan (tuple): (t_start, t_end) time interval.
        y0 (array): Initial conditions for y.
        yp0 (array): Initial conditions for dy/dt.
        dt (float): Time step size.
        max_iter (int): Maximum iterations for Newton's method.
        tol (float): Convergence tolerance.

    Returns:
        t (array): Time points.
        y (array): Solution values.
    """
    t0, tf = tspan
    t_values = np.arange(t0, tf + dt, dt)
    y_values = [y0]
    y = y0.copy()

    for t in t_values[:-1]:
        y_new = y.copy()
        for _ in range(max_iter):
            dydt = (y_new - y) / dt
            residual = f(t + dt, y_new, dydt)
            if np.linalg.norm(residual) < tol:
                break
            jacobian = np.eye(len(y0)) - dt * np.eye(len(y0))
            delta = np.linalg.solve(jacobian, -residual)
            y_new += delta
        y_values.append(y_new)
        y = y_new

    return t_values, np.array(y_values)


def solve_ivp_mass(f, tspan, y0, n=200, mass=None, atol=1e-6, rtol=1e-3, debug=False, method='euler'):
    """
    Solves a differential algebraic equation system with support for mass matrix
        mass*dydt = f(t,y)

    Methods:
      - 'euler'   : adaptive implicit Euler (self-contained)
      - 'bdf2'    : adaptive BDF2 (self-contained)
      - 'sdirk'   : adaptive SDIRK2 (self-contained)
      - 'assimulo': IDA (variable-order BDF) via Assimulo

    Uses root() to solve implicit systems. n controls an initial step heuristic.
    """
    if np.ndim(y0) == 0:
        m = 1
    else:
        m = len(y0)

    if mass is None:
        mass = np.eye(len(y0))

    y0 = np.array(y0)

    f0 = f(0, y0)
    if not isinstance(f0, np.ndarray):
        raise Exception('Derivative must be a np.array')

    aeIdx = np.diag(mass) < 0.5

    if np.sum(aeIdx) < m:
        qinit = y0[aeIdx]
        to = 0.0
        yo = y0
        tfact0 = 1e-10
        tp = tfact0
        yp = y0
        sol = root(lambda q: dae_init_residual(q, aeIdx, yp, f, to, yo, tp, mass), qinit, method='hybr')

        if debug:
            print(f'nfev: {sol["nfev"]}')
        if not sol['success']:
            print('Warning: Failed to initialize dae')
        qopt = sol['x']
        y0[aeIdx] = qopt

    if method == 'assimulo':
        ''' assimulo can be installed with
        
        conda install conda-forge::assimulo
        '''
        from assimulo.solvers import IDA
        from assimulo.problem import Implicit_Problem

        def residual(t, y, yd, f, mass):
            return mass @ yd - f(t, y)

        t0 = tspan[0]
        yd0 = f(t0, y0)
        model = Implicit_Problem(lambda t, y, yd: residual(t, y, yd, f, mass), y0, yd0, t0)

        sim = IDA(model)
        sim.atol = atol
        sim.rtol = rtol
        sim.simulate(tspan[1], n)

        sol = Sol()
        sol.t = np.array(sim.t_sol)
        sol.y = np.array(sim.y_sol).T

    elif method == 'euler':
        def residual(t, y, yd, f, mass):
            return mass @ yd - f(t, y)

        model = lambda t, y, yd: residual(t, y, yd, f, mass)

        t0 = tspan[0]
        tf = tspan[1]
        yd0 = f(t0, y0)

        initial_max_step = min(0.1, (tf - t0) / max(10, n))

        solver = DAESolver(
            model, y0, yd0, t0, tf,
            atol=atol, rtol=rtol,
            max_step=initial_max_step, min_step=1e-9,
            root_method='hybr',
            xtol=atol,
            maxfev=None,
            safety=0.9,
            min_factor=0.2,
            max_factor=5.0,
            reduced=True
        )

        tsol, ysol = solver.solve()
        sol = Sol()
        sol.t = np.array(tsol)
        sol.y = np.array(ysol).T

    elif method == 'bdf2':
        def residual(t, y, yd, f, mass):
            return mass @ yd - f(t, y)

        model = lambda t, y, yd: residual(t, y, yd, f, mass)

        t0 = tspan[0]
        tf = tspan[1]
        yd0 = f(t0, y0)

        initial_max_step = min(0.1, (tf - t0) / max(10, n))

        solver = BDF2DAESolver(
            model, y0, yd0, t0, tf,
            atol=atol, rtol=rtol,
            max_step=initial_max_step, min_step=1e-9,
            root_method='hybr',
            xtol=atol,
            maxfev=None,
            safety=0.9,
            min_factor=0.2,
            max_factor=5.0,
        )

        tsol, ysol = solver.solve()
        sol = Sol()
        sol.t = np.array(tsol)
        sol.y = np.array(ysol).T

    elif method == 'sdirk':
        def residual(t, y, yd, f, mass):
            return mass @ yd - f(t, y)

        model = lambda t, y, yd: residual(t, y, yd, f, mass)

        t0 = tspan[0]
        tf = tspan[1]
        yd0 = f(t0, y0)

        initial_max_step = min(0.1, (tf - t0) / max(10, n))

        solver = SDIRK2DAESolver(
            model, y0, yd0, t0, tf,
            atol=atol, rtol=rtol,
            max_step=initial_max_step, min_step=1e-9,
            root_method='hybr',
            xtol=atol,
            maxfev=None,
            safety=0.9,
            min_factor=0.2,
            max_factor=5.0,
        )

        tsol, ysol = solver.solve()
        sol = Sol()
        sol.t = np.array(tsol)
        sol.y = np.array(ysol).T

    else:
        raise ValueError(f"Unknown method '{method}'")

    return sol


def dae_init_residual(q, aeIdx, yp, f, to, yo, tp, mass):
    yp[aeIdx] = q
    value = backward_euler_residual(yp, f, to, yo, tp, mass)
    resvalue = value[aeIdx]
    return resvalue


def backward_euler_residual(yp, f, to, yo, tp, mass):
    value = mass.dot(yp - yo) - (tp - to) * f(tp, yp)
    return value