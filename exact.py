"""
This program is based on 'https://github.com/chunatho/SodShockSolution.git'
"""
import numpy as np
from scipy.optimize import newton

class SodExactSolution:
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def _root_f(self, P, pL, pR, cL, cR, g, uR, uL):
        # Nonlinear equation for P* (see Toro eqn 4.31)
        a = (g-1)*(cR/cL)*(P-1)
        b = np.sqrt(2*g*(2*g + (g+1)*(P-1)))
        return P - pL/pR * (1 - a/b)**(2*g/(g-1)) + uL - uR

    def solve(self, x, t, left, right, x0=0.5):
        """
        Calculate Sod shock tube analytical solution at time t.
        x: spatial grid (1D array)
        t: time (float)
        left, right: initial state [rho, u, p]
        x0: initial position of the interface (float)
        Returns: rho, u, p, e_int (all arrays)
        """
        g = self.gamma
        rhoL, uL, pL = left
        rhoR, uR, pR = right

        # t=0: simply assign initial conditions
        if t == 0:
            rho = np.where(x < x0, rhoL, rhoR)
            u   = np.where(x < x0, uL, uR)
            p   = np.where(x < x0, pL, pR)
            e_int = p / ((g-1) * rho)
            return rho, u, p, e_int
        
        # Node spacing
        dx = x[1] - x[0]
        Nx = len(x)
        # The index where interface sits (nearest to x0)
        x_idx = np.argmin(np.abs(x - x0))

        # Sound speeds
        cL = np.sqrt(g * pL / rhoL)
        cR = np.sqrt(g * pR / rhoR)

        # Find intermediate pressure P*
        # [Note] Pstar is nondimensionalized by pr. (it's ok using pl).
        #        So if you use Pstar for computing density or velosity, you times pr for Pstar to get dimensional Pstar (Pstar*pR)
        Pstar = newton(self._root_f, 0.5, args=(pL, pR, cL, cR, g, uR, uL), tol=1e-12)
        # Star region velocity (Toro 4.46)
        u_star = uL + 2*cL/(g-1)*(1 - (Pstar*pR/pL)**((g-1)/(2*g))) # uL - fL (Function fL for Left rarefaction)/dimensionalized Pstar
        # Densities in star regions 
        rho3 = rhoL * (Pstar * pR / pL)**(1/g) # dimensionalized Pstar
        p3 = Pstar * pR  # dimensionalized Pstar
        # Shock and fan boundaries
        c_shock = uR + cR * np.sqrt((g-1 + Pstar*(g+1)) / (2*g)) # Toro 4.59
        x_shock = x0 + c_shock * t # Shock position between Region 1 and 2
        c_contact = u_star
        x_contact = x0 + c_contact * t # Contact discontinuity position between Region 2 and 3
        c_leftfan_tail = u_star - np.sqrt(g*p3/rho3)
        x_leftfan_tail = x0 + c_leftfan_tail * t # Position of tail for the left-side rarefaction fan between Region 3 and 4 
        c_leftfan_head = uL - cL
        x_leftfan_head = x0 + c_leftfan_head * t # Position of head for the left-side rarefaction fan between Region 4 and 5

        # Prepare outputs
        rho = np.zeros_like(x)
        u = np.zeros_like(x)
        p = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi < x_leftfan_head:
                # Left state
                rho[i] = rhoL
                u[i] = uL
                p[i] = pL
            elif xi < x_leftfan_tail:
                # Rarefaction fan (left)
                u4 = 2/(g+1) * (cL + (xi-x0)/t)
                rho[i] = rhoL * (1 - (g-1)/2 * u4/cL)**(2/(g-1))
                u[i] = u4
                p[i] = pL * (1 - (g-1)/2 * u4/cL)**(2*g/(g-1))
            elif xi < x_contact:
                # Region 3 (left of contact)
                rho[i] = rho3
                u[i] = u_star
                p[i] = p3
            elif xi < x_shock:
                # Region 2 (right of contact)
                # post-shock density (see Toro eq.4.36)
                rho2 = (g+1)*Pstar*rhoR + (g-1)*rhoR
                rho2 /= (g-1)*Pstar + (g+1)
                u[i] = u_star
                p[i] = p3
                rho[i] = rho2
            else:
                # Right state
                rho[i] = rhoR
                u[i] = uR
                p[i] = pR

        e_int = p / ((g-1)*rho)
        return rho, u, p, e_int
