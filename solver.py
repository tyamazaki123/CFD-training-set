import os
import sys
import configparser
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

import SodExactSolution

class SodShockTubeSolver:
    def __init__(self, config):
        """Initialize solver with parameters from config (ConfigParser section)."""
        # Physical constants
        self.gamma = 1.4
        self.gamma1 = self.gamma - 1.0
        self.gamma1v = 1.0 / self.gamma1
        # Load configuration parameters
        self.nxmax = config.getint("nxmax")
        self.dt = config.getfloat("dt")
        self.loop = config.getint("loop")
        self.nout = config.getint("nout")
        # Optional: tout for logging frequency (default to nout if not provided)
        if "tout" in config:
            self.tout = config.getint("tout")
        else:
            self.tout = self.nout
        # RHS selection
        rhs_opt = str(config.get("rhs")).lower()
        if rhs_opt.isdigit():
            self.rhs_code = int(rhs_opt)
        else:
            # Map scheme name to code
            if rhs_opt in ["-1","noMUSCL"]:
                self.rhs_code = -1
            elif rhs_opt in ["1","muscl","MUSCL","MUSL"]:
                self.rhs_code = 1
            else:
                raise ValueError(f"Unknown high-order scheme: {rhs_opt}")

        # Scheme selection (supports int code or name)
        scheme_opt = str(config.get("scheme")).lower()
        if scheme_opt.isdigit():
            scheme_code = int(scheme_opt)
        else:
            # Map scheme name to code
            if scheme_opt in ["0", "lax", "lax-friedrichs"]:
                scheme_code = 0
            elif scheme_opt in ["1", "lax-wendroff", "lax_wendroff"]:
                scheme_code = 1
            elif scheme_opt in ["2", "roe"]:
                scheme_code = 2
            elif scheme_opt in ["3", "slau"]:
                scheme_code = 3
            elif scheme_opt in ["4", "weno"]:
                scheme_code = 4
            elif scheme_opt in ["5", "kep"]:
                scheme_code = 5
            elif scheme_opt in ["6", "keep"]:
                scheme_code = 6
            else:
                raise ValueError(f"Unknown scheme: {scheme_opt}")
        # Choose scheme class
        if scheme_code == 0:
            self.scheme = LaxScheme()
        elif scheme_code == 1:
            self.scheme = LaxWendroffScheme()
        elif scheme_code == 2:
            self.scheme = RoeScheme()
        elif scheme_code == 3:
            self.scheme = SLAUScheme()
        elif scheme_code == 4:
            self.scheme = WENOscheme()
        elif scheme_code == 5:
            self.scheme = KEPscheme()
        elif scheme_code == 6:
            self.scheme = KEEPscheme()
        else:
            raise ValueError(f"Unsupported scheme code: {scheme_code}")
        # Spatial domain
        self.xlength = config.getfloat("xlength") if "xlength" in config else 1.0
        self.dx = self.xlength / self.nxmax
        # Allocate arrays (using numpy for vectorized operations)
        # +2 for ghost cells at index 0 and nxmax+1
        self.qc = np.zeros((3, self.nxmax + 2))      # [density, momentum, energy]
        self.qc_old = np.zeros_like(self.qc)         # previous step conserved variables
        self.prtv = np.zeros((3, self.nxmax + 2))    # [density, velocity, pressure] primitives
        self.prtvl = np.zeros((3, self.nxmax + 2))    # [density, velocity, pressure] primitives at left side
        self.prtvr = np.zeros((3, self.nxmax + 2))    # [density, velocity, pressure] primitives at right side
        self.dflx = np.zeros((3, self.nxmax + 2))    # flux difference (F_{i+1/2}-F_{i-1/2})/dx
        self.flux = np.zeros((3, self.nxmax + 3))    # flux at interfaces (for Roe, size nxmax+2 interfaces plus one extra index)
        # Initialize time
        self.time = 0.0
        # Set history
        self.history = []
        self.history_exact = []
        self.exact = SodExactSolution(self.gamma)
        self.debug_loop = 0
        
    def compute_primitives(self):
        """Convert conserved variables (qc) to primitive variables (prtv)."""
        # Avoid division by zero by handling density carefully (rho > 0 in valid states)
        rho = self.qc[0]
        u = np.zeros_like(rho)
        # Compute velocity where density is nonzero
        nonzero = rho != 0
        u[nonzero] = self.qc[1, nonzero] / rho[nonzero]
        # Pressure from E, rho, u: p = (γ-1)*(E - 0.5*rho*u^2)
        E = self.qc[2]
        p = self.gamma1 * (E - 0.5 * rho * u**2)
        # Assign to primitive array
        self.prtv[0] = rho
        self.prtv[1] = u
        self.prtv[2] = p
        # Safety check for negative pressure or density (optional)
        if np.any(rho[1:self.nxmax+1] < 0) or np.any(p[1:self.nxmax+1] < 0):
            raise RuntimeError("Negative density or pressure encountered in simulation!")

        self.prtvl = self.prtv
        self.prtvr = self.prtv
        
    def run(self, ini_cond, boundary, rhs, log):
        """Execute the time-stepping simulation."""
        # Apply initial condition
        ini_cond.apply(self)
        # Output initial state
        self.output(step=0)
        # Log initial info
        log.write(f"Loop = {self.loop}\n")
        log.write(f"dx = {self.dx:.2e}\n")
        log.write(f"dt = {self.dt:.2e}\n")
        log.write("Starting simulation...\n")
        # Main time integration loop
        for step in range(1, self.loop + 1):
            if step % self.nout == 0:print("step = ",step)
            # Compute primitives from conserved vars
            self.compute_primitives()
            # Enforce boundary conditions on primitives
            boundary.apply(self.prtv)
            # Enhance the spatial order for conservatives
            rhs.apply(self)
            # Perform one step of the selected scheme
            res = self.scheme.step(self)
            # Update time and previous state
            self.time += self.dt
            self.qc_old[:, :] = self.qc[:, :]
            # Log progress at specified intervals
            if step % self.tout == 0 or step == self.loop:
                log.write(f"step = {step}\n")
                log.write(f"time = {self.time:15.7e}\n")
                log.write(f"res  = {res:15.7e}\n")
            # Output results at specified intervals
            if step % self.nout == 0 or step == self.loop:
                self.save_current_state()
                self.save_current_exact_state
                self.output(step)
        self.output_animation_gif()
        log.write("Simulation complete.\n")


    def save_current_state(self):
        """save physical quantities to history"""
        rho = self.qc[0, 1:self.nxmax+1].copy()
        u = self.qc[1, 1:self.nxmax+1] / rho
        p = self.gamma1 * (self.qc[2, 1:self.nxmax+1] - 0.5 * rho * u**2)
        e = p / (self.gamma1 * rho)
        self.history.append((rho, u, p, e))

    def save_current_exact_state(self):
        # Left side initial state
        left = [1.0,0.0,1.0] # density, velocity, pressure
        right = [0.125,0.0,1.0]
        x0 = self.xlength / 2.0
        rho,u,p,e = self.exact(rho,self.time,left,right,x0)
        
        self.history_exact.append((rho,u,p,e))
        
    def output(self, step):
        """Plot and save the current state (density, velocity, pressure, internal energy) to a PDF file."""
        # Compute primitive variables for plotting (density, velocity, pressure)
        rho = self.qc[0, 1:self.nxmax+1]
        u = self.qc[1, 1:self.nxmax+1] / rho
        p = self.gamma1 * (self.qc[2, 1:self.nxmax+1] - 0.5 * rho * u**2)
        # Specific internal energy e = p / [(γ-1)*ρ]
        e_int = p / (self.gamma1 * rho)
        # Spatial coordinates for cell centers (1..nxmax)
        x = np.linspace(1, self.nxmax, self.nxmax) / self.nxmax * self.xlength
        # Create a 2x2 subplot for the four variables
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs[0, 0].plot(x, rho, color='C0')
        axs[0, 0].set_title("Density")
        axs[0, 0].set_xlabel("Position x")
        axs[0, 0].set_ylabel("Density ρ")
        axs[0, 1].plot(x, u, color='C1')
        axs[0, 1].set_title("Velocity")
        axs[0, 1].set_xlabel("Position x")
        axs[0, 1].set_ylabel("Velocity u")
        axs[1, 0].plot(x, p, color='C2')
        axs[1, 0].set_title("Pressure")
        axs[1, 0].set_xlabel("Position x")
        axs[1, 0].set_ylabel("Pressure p")
        axs[1, 1].plot(x, e_int, color='C3')
        axs[1, 1].set_title("Internal Energy")
        axs[1, 1].set_xlabel("Position x")
        axs[1, 1].set_ylabel("Internal e")
        fig.suptitle(f"Step {step} (Time = {self.time:.4f})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save to PDF in output directory
        fname = f"output/field{step:05d}.pdf"
        plt.savefig(fname, format='pdf')
        plt.close(fig)
        # (No need to explicitly cite the image source in text; it will be in output folder)

    def output_animation_gif(self):
        """export GIF file from saved history"""
        if not self.history:
            return  # No data

        x = np.linspace(1, self.nxmax, self.nxmax) / self.nxmax * self.xlength

        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        lines = []
        colors = ['C0','C1','C2','C3']
        for ax,color in zip(axs.flat,colors):
            line, = ax.plot([], [], lw=2,color=color)
            lines.append(line)
        titles = ["Density", "Velocity", "Pressure", "Internal Energy"]
        ylabels = [r"$\rho$", r"$u$", r"$p$", r"$e$"]
        for ax, title, ylabel in zip(axs.flat, titles, ylabels):
            ax.set_xlim(x[0], x[-1])
            if title == "Internal Energy":
                ax.set_ylim(1.5, 2.9)
            else:
                ax.set_ylim(0,1.1)
            ax.set_title(title)
            ax.set_xlabel("Position x")
            ax.set_ylabel(ylabel)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            rho, u, p, e = self.history[i]
            lines[0].set_data(x, rho)
            lines[1].set_data(x, u)
            lines[2].set_data(x, p)
            lines[3].set_data(x, e)
            fig.suptitle(f"Step {i*self.nout} (Frame {i+1}/{len(self.history)})")
            return lines

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.history), interval=200, blit=True)
        gif_path = "output/sod_shock_tube.gif"
        anim.save(gif_path, writer=PillowWriter(fps=5))
        plt.close(fig)
        print(f"Export animation gif: {gif_path}")
        
class InitialCondition:
    """Initialize the shock tube problem (Sod shock tube initial states)."""
    def apply(self, solver):
        # Halfway index
        mid = solver.nxmax // 2  
        # Loop over all cells (including ghost cells 0 and nxmax+1)
        for j in range(solver.nxmax + 2):
            if j <= mid:
                # Left side initial state
                rho = 1.0    # density
                u   = 0.0    # velocity
                p   = 1.0    # pressure
            else:
                # Right side initial state
                rho = 0.125  # density
                u   = 0.0    # velocity
                p   = 0.1    # pressure
            # Set conserved variables in solver.qc:
            solver.qc[0, j] = rho
            solver.qc[1, j] = rho * u
            solver.qc[2, j] = solver.gamma1v * p + 0.5 * rho * u**2
        # Copy initial state to qc_old
        solver.qc_old[:, :] = solver.qc[:, :]
        solver.time = 0.0  # start time
        
class BoundaryCondition:
    """Apply reflective (wall) boundary conditions: zero velocity at walls."""
    def apply(self, prtv):
        # Zero velocity at the first and last physical cell
        prtv[1, 1] = 0.0             # index 1 = first interior cell's velocity
        prtv[1, -2] = 0.0            # -2 index corresponds to prtv[1, nxmax]
        # Copy boundary cell primitive values into ghost cells
        prtv[:, 0] = prtv[:, 1]      # left ghost = first cell
        prtv[:, -1] = prtv[:, -2]    # right ghost = last cell

class RHS:
    def apply(self,solver):
        if solver.rhs_code == -1: # without any spatial high
            return
        else:
            if solver.rhs_code == 1:
                self.MUSCL(solver)
            else:
                print('Other schems have not been implemented yet.')
                exit()
        return
        
    def MUSCL(self,solver):
        limiter_type = "vanleer"

        nxmax = solver.nxmax
        gamma = solver.gamma
        order = 3
        qc = solver.qc
        dx = solver.dx
        dt = solver.dt

        # MUSCL Interpolation : left state(qL) and right state(qr) are calculated at each interface
        qL = np.zeros_like(solver.qc)  # i+1/2 left side
        qR = np.zeros_like(solver.qc)  # i+1/2 right side

        for m in range(3):
            for i in range(1, nxmax+1):
                dq_minus = qc[m, i  ] - qc[m, i-1]
                dq_plus  = qc[m, i+1] - qc[m, i  ]
                if order == 1:
                    phi_l = 0.0
                    phi_r = 0.0
                    kai = 0.0
                else:
                    # limitter value
                    r_l = (qc[m, i-1] - qc[m,-2]) / (qc[m,i  ] - qc[m,i-1] + 1e-12)
                    r_r = (qc[m, i  ] - qc[m,-1]) / (qc[m,i+1] - qc[m,i  ] + 1e-12)
                    phi_l = self.limiter(r_l, limiter_type)
                    phi_r = self.limiter(r_r, limiter_type)
                    if order == 2:
                        kai = -1.0
                    elif order == 3:
                        kai = 1.0/3.0
                    else:
                        print('MUSCL is first to third order schem.')
                        exit()
                # MUSCL Interpolation
                qL[m, i] = qc[m, i] + 0.25 * phi_l * ((1.0-kai)*dq_minus + (1.0+kai)*dq_plus)
                qR[m, i] = qc[m, i] - 0.25 * phi_r * ((1.0-kai)*dq_plus  + (1.0+kai)*dq_minus)
                
            # first order interpolation at domain boundary
            qL[m,0] = qc[m,0]
            qR[m,0] = qc[m,0]
            qL[m,1] = qc[m,1]
            qR[m,1] = qc[m,1]

            qL[m,-2] = qc[m,-2]
            qR[m,-2] = qc[m,-2]
            qL[m,-1] = qc[m,-1]
            qR[m,-1] = qc[m,-1]

        # if solver.debug_loop == 1: exit()
        # solver.debug_loop += 1
            
        # Compute primitive values
        solver.prtvl = self.compute_intp_primitives(qL,solver)
        solver.prtvr = self.compute_intp_primitives(qR,solver)

        return 
    
    def limiter(self, r, typ="vanleer"):
        if typ == "minmod":
            return np.maximum(0, np.minimum(1, r))
        elif typ == "superbee":
            return np.maximum(0, np.maximum(np.minimum(2*r, 1), np.minimum(r, 2)))
        elif typ == "vanleer":
            return (r + np.abs(r)) / (1 + np.abs(r))
        elif typ == "mc":
            return np.maximum(0, np.minimum((1 + r) / 2, np.minimum(2, 2*r)))
        else:
            # default (vanleer)
            return (r + np.abs(r)) / (1 + np.abs(r))

    def compute_intp_primitives(self,qc,solver):
        """Convert interpolated- and conserved-variables (qc) to interpolated-primitive variables (prtv)."""
        prtv = np.zeros_like(qc)
        # Avoid division by zero by handling density carefully (rho > 0 in valid states)
        rho = qc[0]
        u = np.zeros_like(rho)
        # Compute velocity where density is nonzero
        nonzero = rho != 0
        u[nonzero] = qc[1, nonzero] / rho[nonzero]
        # Pressure from E, rho, u: p = (γ-1)*(E - 0.5*rho*u^2)
        E = qc[2]
        p = solver.gamma1 * (E - 0.5 * rho * u**2)
        # Assign to primitive array
        prtv[0] = rho
        prtv[1] = u
        prtv[2] = p

        # Safety check for negative pressure or density (optional)
        if np.any(rho[1:solver.nxmax+1] < 0) or np.any(p[1:solver.nxmax+1] < 0):
            raise RuntimeError("Negative density or pressure encountered in simulation!")

        return prtv
        
class LaxScheme:
    """Lax-Friedrichs scheme (flux splitting with artificial dissipation)."""
    def step(self, solver):
        nx = solver.nxmax
        # Compute flux at every node (including ghosts) using primitive vars
        # Flux F = [ρu,  ρu^2 + p,  u*(E + p) ]
        rho = solver.prtv[0]; u = solver.prtv[1]; p = solver.prtv[2]
        E = p * solver.gamma1v + 0.5 * rho * u**2
        # Compute flux arrays for all indices 0..nx+1
        F0 = rho * u
        F1 = rho * u**2 + p
        F2 = (E + p) * u
        # Compute flux divergence for each interior cell j = 1..nx
        solver.dflx[:, 1:nx+1] = (np.array([F0, F1, F2])[:, 2:nx+2] - np.array([F0, F1, F2])[:, 0:nx]) / (2 * solver.dx)
        # Update conserved variables: Q_new = 0.5*(Q[i-1] + Q[i+1]) - dt * (flux_diff)
        # Using solver.qc_old (previous step state) for neighbor values
        solver.qc[:, 1:nx+1] = 0.5 * (solver.qc_old[:, 0:nx] + solver.qc_old[:, 2:nx+2]) - solver.dt * solver.dflx[:, 1:nx+1]
        # Compute max residual (e.g., density change)
        res = np.max(np.abs(solver.dflx[0, 1:nx+1]))
        return res
        
class LaxWendroffScheme:
    """Lax-Wendroff scheme (second-order, using flux Jacobian)."""
    def step(self, solver):
        nx = solver.nxmax
        γ = solver.gamma; γ1 = solver.gamma1
        # Arrays for flux and flux Jacobian (A matrix) at cell faces
        F = np.zeros((3, nx+2))
        A = np.zeros((3, 3, nx+2))
        rho = solver.prtv[0]; u = solver.prtv[1]; p = solver.prtv[2]
        E = p * solver.gamma1v + 0.5 * rho * u**2
        # Compute flux at each cell (including ghosts) for current time
        F[0] = rho * u
        F[1] = rho * u**2 + p
        F[2] = (E + p) * u
        # Compute linearized flux Jacobian A at each half-step (interface) j = 1..nx+1
        for j in range(1, nx+2):
            # Left state (j-1) and right state (j)
            rho_L, u_L, p_L = rho[j-1], u[j-1], p[j-1]
            rho_R, u_R, p_R = rho[j], u[j], p[j]
            # Average state at interface (midpoint)
            rho_half = 0.5 * (rho_L + rho_R)
            u_half   = 0.5 * (u_L + u_R)
            # Internal energy at midpoint (E = p/(γ-1) + 0.5*rho*u^2)
            E_L = p_L * solver.gamma1v + 0.5 * rho_L * u_L**2
            E_R = p_R * solver.gamma1v + 0.5 * rho_R * u_R**2
            E_half = 0.5 * (E_L + E_R)
            # Compute flux Jacobian matrix A = dF/dQ at the half-step state
            # (Euler equations' Jacobian in primitive variables form)
            m = rho_half * u_half  # momentum density at midpoint
            A[:, :, j] = [[0.0,           1.0,                0.0],
                          [0.5*(γ-3)*m**2/(rho_half**2),  -(γ-3)*m/ rho_half,    γ1       ],
                          [-γ*m*E_half/(rho_half**2) + γ1*m**3/(rho_half**3),
                            γ*E_half/ rho_half - 1.5*γ1*m**2/(rho_half**2),      γ * m/ rho_half]]
        # Compute flux divergence with correction terms for each cell j = 1..nx
        for j in range(1, nx+1):
            # Central difference of flux
            dF_center = (F[:, j+1] - F[:, j-1]) / (2 * solver.dx)
            # Forward and backward differences
            dF_forward = F[:, j+1] - F[:, j]
            dF_backward = F[:, j] - F[:, j-1]
            # Flux correction using Jacobians at interfaces
            term_forward = A[:, :, j+1].dot(dF_forward)
            term_backward = A[:, :, j].dot(dF_backward)
            solver.dflx[:, j] = dF_center - (solver.dt / (2 * solver.dx**2)) * (term_forward - term_backward)
        # Update conserved variables: Q_new = Q_old - dt * dF (with Lax-Wendroff corrections)
        solver.qc[:, 1:nx+1] = solver.qc[:, 1:nx+1] - solver.dt * solver.dflx[:, 1:nx+1]
        # Max residual (density change)
        res = np.max(np.abs(solver.dflx[0, 1:nx+1]))
        return res

        
class RoeScheme:
    """Roe approximate Riemann solver scheme."""
    def step(self, solver):
        nx = solver.nxmax
        γ = solver.gamma; γ1 = solver.gamma1
        # Compute flux at each interface (face) j = 1..nx+1
        for j in range(1, nx+2):
            # Left (L) and Right (R) primitive states at the interface
            rL = solver.prtvl[0, j-1]; uL = solver.prtvl[1, j-1]; pL = solver.prtvl[2, j-1]
            rR = solver.prtvr[0, j];   uR = solver.prtvr[1, j];   pR = solver.prtvr[2, j]

            # sound speeds
            cL = np.sqrt(γ * pL / rL)
            cR = np.sqrt(γ * pR / rR)
            # Enthalpies H = (E + p)/ρ = c^2/(γ-1) + 0.5*u^2
            HL = cL**2 * solver.gamma1v + 0.5 * uL**2
            HR = cR**2 * solver.gamma1v + 0.5 * uR**2
            # Roe average state
            sqrt_rL = np.sqrt(rL); sqrt_rR = np.sqrt(rR)
            # Weighted average of velocity and enthalpy
            u_t = (sqrt_rL * uL + sqrt_rR * uR) / (sqrt_rL + sqrt_rR)
            H_t = (sqrt_rL * HL + sqrt_rR * HR) / (sqrt_rL + sqrt_rR)
            # Roe-averaged sound speed
            c_t = np.sqrt(max(0.0, (γ-1) * (H_t - 0.5 * u_t**2)))
            # Compute differences in conserved variables across the interface
            # (We'll use primitive differences for calculating wave strengths)
            dr = rR - rL
            du = uR - uL
            dp = pR - pL
            # Wave strength (Roe) coefficients (alpha values)
            # Contact wave (lambda1 = u_t)
            alpha1 = dr - dp / (c_t**2)
            # Right-moving acoustic wave (lambda2 = u_t + c_t)
            alpha2 = (dp + (r_t := sqrt_rL * sqrt_rR) * c_t * du) / (2 * c_t**2)
            # Left-moving acoustic wave (lambda3 = u_t - c_t)
            alpha3 = (dp - r_t * c_t * du) / (2 * c_t**2)
            # Eigenvalues (characteristic speeds)
            λ1 = u_t
            λ2 = u_t + c_t
            λ3 = u_t - c_t
            # Absolute values for upwinding (Roe's dissipation)
            a1, a2, a3 = abs(λ1), abs(λ2), abs(λ3)
            # Right eigenvectors of Roe matrix (columns):
            # Contact (entropy) wave, and acoustic waves
            r_contact = np.array([1.0, u_t, 0.5 * u_t**2])
            r_right   = np.array([1.0, u_t + c_t, H_t + u_t * c_t])
            r_left    = np.array([1.0, u_t - c_t, H_t - u_t * c_t])
            # Compute left and right physical fluxes (F(Q_L) and F(Q_R))
            EL = pL * solver.gamma1v + 0.5 * rL * uL**2  # total energy (left)
            ER = pR * solver.gamma1v + 0.5 * rR * uR**2  # total energy (right)
            F_L = np.array([rL * uL, 
                            rL * uL**2 + pL, 
                            (EL + pL) * uL])
            F_R = np.array([rR * uR, 
                            rR * uR**2 + pR, 
                            (ER + pR) * uR])
            # Roe flux: F_face = 0.5*(F_L + F_R) - 0.5 * \sum_i (alpha_i * |λ_i| * r_i)
            flux_face = 0.5 * (F_L + F_R) - 0.5 * (alpha1 * a1 * r_contact 
                                                  + alpha2 * a2 * r_right 
                                                  + alpha3 * a3 * r_left)
            # Store flux at this interface
            solver.flux[:, j] = flux_face

        # Compute flux divergence for each cell j = 1..nx: (F_face_right - F_face_left)/dx
        solver.dflx[:, 1:nx+1] = (solver.flux[:, 2:nx+2] - solver.flux[:, 1:nx+1]) / solver.dx
        # Update conserved variables: Q_new = Q_old - dt * (flux_difference)
        solver.qc[:, 1:nx+1] = solver.qc_old[:, 1:nx+1] - solver.dt * solver.dflx[:, 1:nx+1]
        # Max residual (density)
        res = np.max(np.abs(solver.dflx[0, 1:nx+1]))
        return res



class SLAUScheme:
    """
    SLAU (Simple Low-dissipation AUSM) scheme for 1D Euler equations (Shima & Kitamura, JCP 2011).
    """
    def step(self, solver):
        nx = solver.nxmax
        gamma = solver.gamma
        dx = solver.dx
        dt = solver.dt
        qc = solver.qc

        # 保存量の新しい配列
        qc_new = np.zeros_like(qc)
        flux = np.zeros((3, nx + 1))

        # ゴーストセルを含めた状態で数値流束を計算
        for i in range(nx + 1):
            # 左・右セル
            rho_L = qc[0, i]
            u_L = qc[1, i] / rho_L
            E_L = qc[2, i]
            p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L**2)
            HL = (E_L + p_L) / rho_L
            a_L = np.sqrt(gamma * p_L / rho_L)

            rho_R = qc[0, i+1]
            u_R = qc[1, i+1] / rho_R
            E_R = qc[2, i+1]
            p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R**2)
            HR = (E_R + p_R) / rho_R
            a_R = np.sqrt(gamma * p_R / rho_R)

            # 基準音速
            a_face = min(a_L, a_R)

            # マッハ数
            M_L = u_L / a_face
            M_R = u_R / a_face

            # SLAU補間: マッハ関数
            def M_plus(M):   # for 0 ≦ M
                return 0.25*(M+1)**2   if abs(M) < 1 else max(M, 0)
            def M_minus(M): # for M ≦ 0
                return -0.25*(M-1)**2  if abs(M) < 1 else min(M, 0)

            # Convective mass flux (Shima 2011 eqn. (23))
            mass_flux = a_face * (rho_L * M_plus(M_L) + rho_R * M_minus(M_R))

            # Pressure flux: (Shima 2011 eqn. (27))
            alpha = 0.1875  # 論文推奨値（可調整）
            beta = 0.125
            f_p = 0.5 * (1 + np.sign(M_L) * (1 - 2 * alpha * M_L * M_R))
            pressure_flux = f_p * p_L + (1 - f_p) * p_R

            # 付加項: 圧力のスムージング
            phi = 0.5 * (1 + np.tanh(beta * (M_L - M_R)))
            pressure_flux = phi * p_L + (1 - phi) * p_R

            # 保存量の数値流束
            # F = [mass, momentum, energy]
            flux[0, i] = mass_flux
            flux[1, i] = mass_flux * ((u_L if mass_flux > 0 else u_R)) + pressure_flux
            flux[2, i] = mass_flux * ((HL if mass_flux > 0 else HR))

        # 保存量の更新（内点のみ、境界は外部で処理）
        for j in range(1, nx + 1):
            qc_new[:, j] = qc[:, j] - dt/dx * (flux[:, j] - flux[:, j-1])

        # 更新
        solver.qc[:, 1:nx+1] = qc_new[:, 1:nx+1]
        res = np.linalg.norm(qc_new - qc)
        return res


class WENOScheme:
    """
    5次精度WENOスキームによる数値流束
    SodShockTubeSolverクラスで scheme_code=4 で呼ばれる
    """
    def step(self, solver):
        # 変数の省略形
        nx = solver.nxmax
        gamma = solver.gamma

        # 保存変数配列
        qc = solver.qc
        dt = solver.dt
        dx = solver.dx

        # 新しい保存変数（出力用）
        qc_new = np.zeros_like(qc)

        # 各成分（ρ, ρu, E）ごとにWENO補間で流束Fを計算
        for k in range(3):
            F = self.compute_flux(qc, k, gamma)
            F_face = self.weno5_reconstruction(F)

            # 半陰的ループ（i=2からi=nx-1まで：ゴーストセル考慮）
            for i in range(2, nx):
                qc_new[k, i] = qc[k, i] - dt/dx * (F_face[i] - F_face[i-1])

        # 更新
        solver.qc = qc_new
        return np.linalg.norm(qc_new - qc)

    def compute_flux(self, qc, k, gamma):
        """保存変数から流束計算"""
        rho = qc[0, :]
        u = qc[1, :] / rho
        E = qc[2, :]
        p = (gamma - 1.0) * (E - 0.5 * rho * u ** 2)
        F = np.zeros_like(rho)

        if k == 0:
            F = qc[1, :]  # ρu
        elif k == 1:
            F = qc[1, :] * u + p
        elif k == 2:
            F = (E + p) * u
        return F

    def weno5_reconstruction(self, F):
        """WENO5による界面流束の再構成（一次元, L→R）"""
        eps = 1e-6
        nx = len(F)
        F_face = np.zeros(nx)

        # i-1, i, i+1, i+2, i+3が使えるよう2セル余分にとる
        for i in range(2, nx-2):
            # 3つの候補Stencil
            f1 = (2*F[i-2] - 7*F[i-1] + 11*F[i]) / 6.0
            f2 = (-F[i-1] + 5*F[i] + 2*F[i+1]) / 6.0
            f3 = (2*F[i] + 5*F[i+1] - F[i+2]) / 6.0

            # 平滑化指標
            b1 = 13/12 * (F[i-2] - 2*F[i-1] + F[i])**2 + 1/4 * (F[i-2] - 4*F[i-1] + 3*F[i])**2
            b2 = 13/12 * (F[i-1] - 2*F[i] + F[i+1])**2 + 1/4 * (F[i-1] - F[i+1])**2
            b3 = 13/12 * (F[i] - 2*F[i+1] + F[i+2])**2 + 1/4 * (3*F[i] - 4*F[i+1] + F[i+2])**2

            # 非線形重み
            a1 = 0.1 / (eps + b1)**2
            a2 = 0.6 / (eps + b2)**2
            a3 = 0.3 / (eps + b3)**2
            wsum = a1 + a2 + a3
            w1 = a1 / wsum
            w2 = a2 / wsum
            w3 = a3 / wsum

            # WENO再構成
            F_face[i] = w1 * f1 + w2 * f2 + w3 * f3

        # 両端は単純コピー
        F_face[:2] = F[:2]
        F_face[-2:] = F[-2:]
        return F_face


class KEPScheme:
    """
    厳密なKinetic Energy Preserving (KEP) スキーム（Jameson 2008, 2012に基づく）
    スキュー対称中心差分によるオイラー方程式の離散化
    """

    def step(self, solver):
        nx = solver.nxmax
        gamma = solver.gamma
        qc = solver.qc
        dt = solver.dt
        dx = solver.dx

        qc_new = np.zeros_like(qc)
        # ゴーストセル考慮：1～nxまで更新
        for i in range(1, nx+1):
            # 中心差分で数値流束を計算
            fluxR = self.euler_flux(qc[:, i+1], gamma)
            fluxL = self.euler_flux(qc[:, i-1], gamma)
            fluxC = self.euler_flux(qc[:, i],   gamma)

            # Jameson型スキュー対称形式
            # F_{i+1/2} + F_{i-1/2} - F_i のように組み合わせる
            for k in range(3):
                # KEP: 0.25*[F(Q_{i+1}) + F(Q_i)] - 0.25*[F(Q_{i}) + F(Q_{i-1})]
                dF = 0.25*(self.euler_flux(qc[:,i+1],gamma)[k] + self.euler_flux(qc[:,i],gamma)[k]) \
                    - 0.25*(self.euler_flux(qc[:,i],gamma)[k] + self.euler_flux(qc[:,i-1],gamma)[k])
                qc_new[k, i] = qc[k, i] - dt/dx * dF

        # 境界（ゴースト）セルはそのまま
        qc_new[:, 0] = qc[:, 0]
        qc_new[:, nx+1] = qc[:, nx+1]
        solver.qc = qc_new
        return np.linalg.norm(qc_new - qc)

    @staticmethod
    def euler_flux(Q, gamma):
        """保存変数 Q = [rho, rho*u, E] から流束 [rho*u, rho*u^2 + p, u*(E+p)] を返す"""
        rho = Q[0]
        u = Q[1] / rho
        E = Q[2]
        p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
        return np.array([
            Q[1],               # rho*u
            Q[1] * u + p,       # rho*u^2 + p
            (E + p) * u         # u*(E + p)
        ])


import numpy as np

class KEEPScheme:
    """
    Kuya & Kawai (2023) KEEP scheme for 1D Euler equations.
    """
    def step(self, solver):
        nxmax = solver.nxmax
        gamma = solver.gamma
        qc = solver.qc  # shape: (3, nxmax+2), includes ghost cells
        
        # Prepare fluxes at cell interfaces (nxmax+1 interfaces)
        flux = np.zeros((3, nxmax + 1))
        
        # For each interface i+1/2, compute the KEEP flux
        for i in range(nxmax + 1):
            qL = qc[:, i]     # left state
            qR = qc[:, i+1]   # right state
            
            # Primitives
            rhoL, momL, EL = qL
            rhoR, momR, ER = qR
            
            uL = momL / rhoL
            uR = momR / rhoR
            pL = (gamma - 1) * (EL - 0.5 * rhoL * uL ** 2)
            pR = (gamma - 1) * (ER - 0.5 * rhoR * uR ** 2)
            
            # Arithmetic average
            avg = lambda a, b: 0.5 * (a + b)
            
            # Density-weighted average velocity
            u_tilde = (rhoL * uL + rhoR * uR) / (rhoL + rhoR)
            
            # Central (KEEP) fluxes
            f1 = avg(rhoL * uL, rhoR * uR)
            f2 = f1 * u_tilde + avg(pL, pR)
            f3 = avg(EL + pL, ER + pR) * u_tilde
            
            # Eigenvalue (for dissipation)
            cL = np.sqrt(gamma * pL / rhoL)
            cR = np.sqrt(gamma * pR / rhoR)
            lamb = np.abs(u_tilde) + max(cL, cR)
            
            dq = qR - qL
            D = 0.5 * lamb * dq
            
            # Final flux (central - dissipation)
            flux[:, i] = np.array([f1, f2, f3]) - D
        
        # Update conserved variables using finite volume
        # qc[:, j] = qc_old - (dt/dx) * (flux_{j+1/2} - flux_{j-1/2})
        res = 0.0
        for j in range(1, nxmax+1):
            dF = flux[:, j] - flux[:, j-1]
            solver.qc[:, j] = solver.qc_old[:, j] - (solver.dt / solver.dx) * dF
            res += np.sum(np.abs(dF))
        return res
