import os
import sys
import configparser
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter

from exact import SodExactSolution

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
            elif rhs_opt in ["2","weno","WENO"]:
                self.rhs_code = 2
            elif rhs_opt in ["3","kep","KEP"]:
                self.rhs_code = 3
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
            elif scheme_opt in ["4", "kep"]:
                scheme_code = 4
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
            if self.rhs_code == 3:
                self.scheme = KEPScheme()
            else:
                raise ValueError(f"rhs must set to be KEP (3).")
        else:
            raise ValueError(f"Unsupported scheme code: {scheme_code}")

        # Choose time integration
        lhs_opt = str(config.get("lhs")).lower()
        if lhs_opt.isdigit():
            self.lhs_type = int(lhs_opt)
        else:
            # Map scheme name to code
            if scheme_opt in ["1", "rk1", "RK1" "euler"]:
                self.lhs_type = 1
            elif scheme_opt in ["2", "rk2", "RK2"]:
                self.lhs_type = 2
            elif scheme_opt in ["3", "rk3", "RK3"]:
                self.lhs_type = 3
            else:
                raise ValueError(f"Unknown scheme: {lhs_type}")

    # ... output, logging, etc ...
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
        self.rho_exact = np.zeros(self.nxmax)
        self.u_exact  = np.zeros(self.nxmax)
        self.p_exact = np.zeros(self.nxmax)
        self.e_int_exact = np.zeros(self.nxmax)

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
        neg_rho_idx = np.where(rho[1:self.nxmax+1] <= 0)[0]
        neg_p_idx   = np.where(p[1:self.nxmax+1] <= 0)[0]

        if len(neg_rho_idx) > 0 or len(neg_p_idx) > 0:
            if len(neg_rho_idx) > 0:
                print("Negative density at cells:", neg_rho_idx + 1, "values:", rho[1:self.nxmax+1][neg_rho_idx])
            if len(neg_p_idx) > 0:
                print("Negative pressure at cells:", neg_p_idx + 1, "values:", p[1:self.nxmax+1][neg_p_idx])
            raise RuntimeError("Negative density or pressure encountered in simulation!")

        self.prtvl = self.prtv
        self.prtvr = self.prtv

    def check_cfl(self,qc, gamma, dx, dt):
        """
        Check CFL number and print a warning if dt exceeds CFL limit.
        """
        rho = qc[0]
        u = qc[1] / rho
        E = qc[2]
        p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
        a = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + a)
        cfl = max_speed * dt / dx
        if cfl > 1.0:
            print(f"Warning: CFL number too large ({cfl:.3f} > 1.0). Simulation may be unstable.")
        else:
            print(f"CFL number = {cfl:.3f}")
        
    def run(self, ini_cond, boundary, rhs, lhs, log):
        """Execute the time-stepping simulation."""
        # Apply initial condition
        ini_cond.apply(self)
        # Check CFL
        self.check_cfl(self.qc, self.gamma, self.dx, self.dt)
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
            if self.lhs_type == 1:
                res = lhs.tvd_rk1(self, self.scheme)
            elif self.lhs_type == 2:
                res = lhs.tvd_rk2(self, self.scheme)
            elif self.lhs_type == 3:
                res = lhs.tvd_rk3(self, self.scheme)
            else:
                raise ValueError("Invalid lhs type (should be 1/2/3)")

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
                self.save_current_exact_state()
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
        right = [0.125,0.0,0.1]
        x0 = self.xlength / 2.0
        dx = self.xlength / self.nxmax
        x = np.arange(0.0, self.xlength, dx)
        
        rho,u,p,e_int = self.exact.solve(x,self.time,left,right,x0)

        self.rho_exact = rho
        self.u_exact  = u
        self.p_exact = p
        self.e_int_exact = e_int
    
        self.history_exact.append((rho,u,p,e_int))
        
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
        axs[0, 0].plot(x, rho, color='C0',label='CFD')
        axs[0, 0].set_title("Density")
        axs[0, 0].set_xlabel("Position x")
        axs[0, 0].set_ylabel("Density ρ")
        axs[0, 1].plot(x, u, color='C1',label='CFD')
        axs[0, 1].set_title("Velocity")
        axs[0, 1].set_xlabel("Position x")
        axs[0, 1].set_ylabel("Velocity u")
        axs[1, 0].plot(x, p, color='C2',label='CFD')
        axs[1, 0].set_title("Pressure")
        axs[1, 0].set_xlabel("Position x")
        axs[1, 0].set_ylabel("Pressure p")
        axs[1, 1].plot(x, e_int, color='C3',label='CFD')
        axs[1, 1].set_title("Internal Energy")
        axs[1, 1].set_xlabel("Position x")
        axs[1, 1].set_ylabel("Internal energy e")

        # Analytical solution
        axs[0, 0].plot(x, self.rho_exact, color='C0',ls='dotted',lw=2.0,label='Analysis')
        axs[0, 1].plot(x, self.u_exact, color='C1',ls='dotted',lw=2.0,label='Analysis')
        axs[1, 0].plot(x, self.p_exact, color='C2',ls='dotted',lw=2.0,label='Analysis')
        axs[1, 1].plot(x, self.e_int_exact, color='C3',ls='dotted',lw=2.0,label='Analysis')
        axs[0, 0].legend()
        axs[0, 1].legend()
        axs[1, 0].legend()
        axs[1, 1].legend()
        
        fig.suptitle(f"Step {step} (Time = {self.time:.4f})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save to PDF in output directory
        fname = f"output/field{step:05d}.pdf"
        plt.savefig(fname, format='pdf')
        plt.close(fig)
        # (No need to explicitly cite the image source in text; it will be in output folder)

    def output_animation_gif(self):
        """Export GIF file from saved history (CFD+Analysis)"""
        if not self.history:
            return  # No data

        x = np.linspace(1, self.nxmax, self.nxmax) / self.nxmax * self.xlength

        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        lines = []
        lines_exact = []
        colors = ['C0','C1','C2','C3']
        for ax,color in zip(axs.flat,colors):
            # CFD
            line, = ax.plot([], [], lw=2,color=color,label="CFD")
            lines.append(line)
            # Analysis
            line_ex, = ax.plot([], [], lw=2,color=color,ls='dotted',label="Analysis")
            lines_exact.append(line_ex)

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
            ax.legend()  # show legend

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
        def init():
            for line in lines + lines_exact:
                line.set_data([], [])
            return lines + lines_exact

        def animate(i):
            # CFD
            rho, u, p, e = self.history[i]
            lines[0].set_data(x, rho)
            lines[1].set_data(x, u)
            lines[2].set_data(x, p)
            lines[3].set_data(x, e)
            # Analysis
            rho_ex, u_ex, p_ex, e_ex = self.history_exact[i]
            lines_exact[0].set_data(x, rho_ex)
            lines_exact[1].set_data(x, u_ex)
            lines_exact[2].set_data(x, p_ex)
            lines_exact[3].set_data(x, e_ex)
            fig.suptitle(f"Step {i*self.nout} (Frame {i+1}/{len(self.history)})")

            return lines + lines_exact

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.history), interval=200, blit=True)
        gif_path = "output/sod_shock_tube.gif"
        anim.save(gif_path, writer=PillowWriter(fps=5))
        plt.close(fig)
        print(f"Export animation gif: {gif_path}")

        # save MP4
        mp4_path = "output/sod_shock_tube.mp4"
        try:
            anim.save(mp4_path, writer=FFMpegWriter(fps=5))
            print(f"Export animation mp4: {mp4_path}")
        except Exception as e:
            print(f"MP4 export failed: {e}")
        plt.close(fig)
        
       
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

        # Exact
        solver.rho_exact = solver.qc[0, 1:-1]
        solver.u_exact  = solver.qc[1, 1:-1] / solver.rho_exact
        solver.p_exact =  solver.gamma1 * (solver.qc[2, 1:-1] - 0.5 * solver.rho_exact * solver.u_exact**2)
        solver.e_int_exact = solver.p_exact * solver.gamma1v / solver.qc[0, 1:-1]
        
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
            elif solver.rhs_code == 2:
                self.WENO(solver)
            elif solver.rhs_code == 3: #KEP
                return
            else:
                print('Other schems have not been implemented yet.')
                exit()
        return
        
    def MUSCL(self,solver):
        limiter_type = "minmod"

        nxmax = solver.nxmax
        gamma = solver.gamma
        order = 2
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
            
        # Compute primitive values
        solver.prtvl = self.compute_intp_primitives(qL,solver)
        solver.prtvr = self.compute_intp_primitives(qR,solver)

        return 
    
    def limiter(self, r, typ="vanleer"): # For MUSCL
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

    def WENO(self, solver):
        nxmax = solver.nxmax
        qc = solver.qc

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

        qL, qR = self.weno_characteristic_reconstruct_interface(qc,solver.gamma)
        solver.prtvl = self.compute_intp_primitives(qL, solver)
        solver.prtvr = self.compute_intp_primitives(qR, solver)

        return

    def weno_characteristic_reconstruct_interface(self, qc, gamma):
        """
        特性分解付きWENO界面再構成
        qc: (3, N) 保存変数 [rho, rho*u, E]
        gamma: 比熱比
        返り値: qL, qR  (どちらも (3, N) で界面値が入る)
        """
        N = qc.shape[1]
        qL = np.zeros_like(qc)
        qR = np.zeros_like(qc)

        # 2セル分 ghost cell 必須
        for i in range(2, N-2):
            # ステンシル(5点)を取り出し (各点: [rho, mom, ene])
            stencil = qc[:, i-2:i+3]  # shape (3, 5)

            # 平均状態（Roe平均が理想。ここでは簡単に算術平均でも可）
            rho_mean = np.mean(stencil[0, :])
            u_mean   = np.mean(stencil[1, :] / stencil[0, :])
            E_mean   = np.mean(stencil[2, :])
            p_mean   = (gamma-1) * (E_mean - 0.5 * rho_mean * u_mean**2)
            H_mean   = (E_mean + p_mean) / rho_mean
            a_mean   = np.sqrt(gamma * p_mean / rho_mean)

            # 1D Eulerの右・左固有ベクトル
            R = np.array([
                [1,         1,        1],
                [u_mean-a_mean, u_mean, u_mean+a_mean],
                [H_mean-u_mean*a_mean, 0.5*u_mean**2, H_mean+u_mean*a_mean]
            ])
            # 逆行列で左固有ベクトル
            L = np.linalg.inv(R)

            # ステンシル5点を特性空間へ変換
            w_stencil = L @ stencil  # (3,5)

            # 各特性成分ごとにWENO再構成
            wL = np.zeros(3)
            wR = np.zeros(3)
            for m in range(3):
                fL, fR = self.weno5_reconstruct_interface(w_stencil[m, :])
                wL[m] = fL[3]  # i+1/2 左側
                wR[m] = fR[3]  # i+1/2 右側

            # 保存変数空間に逆変換
            qL[:, i+1] = R @ wL
            qR[:, i+1] = R @ wR

        # 境界部は元の値をそのまま（ghost cell運用必須）
        qL[:, :3] = qc[:, :3]
        qL[:, -3:] = qc[:, -3:]
        qR[:, :3] = qc[:, :3]
        qR[:, -3:] = qc[:, -3:]
        return qL, qR

    def weno5_reconstruct_interface(self, f):
        """
        WENO5 left/right interface reconstruction for a 1D array f.
        Returns fL, fR: left and right states at each interface.
        """
        N = len(f)
        fL = np.zeros(N)
        fR = np.zeros(N)
        eps = 1e-6

        # Linear weights
        g1, g2, g3 = 0.1, 0.6, 0.3

        # Left-biased stencil for fL at i+1/2^-
        for i in range(2, N-2):
            # Left (i+1/2^-)
            P1 = (2*f[i-2] - 7*f[i-1] + 11*f[i]) / 6.0
            P2 = (-f[i-1] + 5*f[i] + 2*f[i+1]) / 6.0
            P3 = (2*f[i] + 5*f[i+1] - f[i+2]) / 6.0

            B1 = (13/12)*(f[i-2] - 2*f[i-1] + f[i])**2 + 0.25*(f[i-2] - 4*f[i-1] + 3*f[i])**2
            B2 = (13/12)*(f[i-1] - 2*f[i] + f[i+1])**2 + 0.25*(f[i-1] - f[i+1])**2
            B3 = (13/12)*(f[i] - 2*f[i+1] + f[i+2])**2 + 0.25*(3*f[i] - 4*f[i+1] + f[i+2])**2

            a1 = g1 / (eps + B1)**2
            a2 = g2 / (eps + B2)**2
            a3 = g3 / (eps + B3)**2
            wsum = a1 + a2 + a3
            w1 = a1 / wsum
            w2 = a2 / wsum
            w3 = a3 / wsum

            fL[i+1] = w1*P1 + w2*P2 + w3*P3

            # Right (i+1/2^+): stencil mirrored
            P1r = (2*f[i+2] - 7*f[i+1] + 11*f[i]) / 6.0
            P2r = (-f[i+1] + 5*f[i] + 2*f[i-1]) / 6.0
            P3r = (2*f[i] + 5*f[i-1] - f[i-2]) / 6.0

            B1r = (13/12)*(f[i+2] - 2*f[i+1] + f[i])**2 + 0.25*(f[i+2] - 4*f[i+1] + 3*f[i])**2
            B2r = (13/12)*(f[i+1] - 2*f[i] + f[i-1])**2 + 0.25*(f[i+1] - f[i-1])**2
            B3r = (13/12)*(f[i] - 2*f[i-1] + f[i-2])**2 + 0.25*(3*f[i] - 4*f[i-1] + f[i-2])**2

            a1r = g1 / (eps + B3r)**2
            a2r = g2 / (eps + B2r)**2
            a3r = g3 / (eps + B1r)**2
            wsumr = a1r + a2r + a3r
            w1r = a1r / wsumr
            w2r = a2r / wsumr
            w3r = a3r / wsumr

            fR[i+1] = w1r*P3r + w2r*P2r + w3r*P1r  # 注: P3r,P2r,P1rの順

        # Boundary: copy values or use lower order
        fL[:3] = f[:3]
        fL[-3:] = f[-3:]
        fR[:3] = f[:3]
        fR[-3:] = f[-3:]

        return fL, fR
        
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
            print(rho)
            print(p)
            raise RuntimeError("Negative density or pressure encountered in simulation!")
        return prtv

class LaxScheme:
    """Lax-Friedrichs scheme (interface-flux form, compatible with Roe/SLAU interface)."""
    def step(self, solver):
        nx = solver.nxmax
        gamma = solver.gamma
        gamma1v = solver.gamma1v
        dx = solver.dx
        dt = solver.dt

        # prtvl and prtvr: primitive variables at left and right of each interface
        prtvl = solver.prtvl  # shape (3, nx+2)
        prtvr = solver.prtvr  # shape (3, nx+2)

        qc = solver.qc
        flux = np.zeros((3, nx+2))

        # Compute local wave speed at each interface
        rL = prtvl[0]; uL = prtvl[1]; pL = prtvl[2]
        rR = prtvr[0]; uR = prtvr[1]; pR = prtvr[2]

        aL = np.sqrt(gamma * pL / rL)
        aR = np.sqrt(gamma * pR / rR)

        alpha = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)  # maximum signal speed at each face

        # Compute fluxes at each interface (face)
        for j in range(1,nx+2):
            # Left and right states at the interface
            rL, uL, pL = prtvl[:, j-1]
            rR, uR, pR = prtvr[:, j]

            # Total energy for left and right
            EL = pL * gamma1v + 0.5 * rL * uL ** 2
            ER = pR * gamma1v + 0.5 * rR * uR ** 2

            # Conserved variables for left and right
            qL = np.array([rL, rL * uL, EL])
            qR = np.array([rR, rR * uR, ER])

            # Physical fluxes for left and right
            FL = np.array([rL * uL, rL * uL ** 2 + pL, (EL + pL) * uL])
            FR = np.array([rR * uR, rR * uR ** 2 + pR, (ER + pR) * uR])

            # Lax-Friedrichs numerical flux at the interface
            flux[:, j] = 0.5 * (FL + FR) - 0.5 * alpha[j] * (qR - qL)

        # Flux difference for each cell (conservative update)
        solver.dflx[:, 0:nx+1] = (flux[:, 1:nx+2] - flux[:, 0:nx+1]) / dx

        # Update conserved variables in all interior cells (Rusanov type)
        solver.qc[:, 1:nx+1] = solver.qc_old[:, 1:nx+1] - dt * solver.dflx[:, 1:nx+1]

        # Return maximum density residual as convergence measure
        res = np.max(np.abs(solver.dflx[0, 0:nx+1]))
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
            rho_L, u_L, pL = rho[j-1], u[j-1], p[j-1]
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
            # Total Enthalpies H = (E + p)/ρ=c^2/(γ-1) + 0.5*u^2
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
            alpha2 = (dp + (r_t := sqrt_rL * sqrt_rR) * c_t * du) / (2 * c_t**2) # walrus operator (over python3.8)
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
    SLAU (Simple Low-dissipation AUSM) scheme for 1D Euler equations.
    Based on: Shima & Kitamura, AIAA 2009-136, JCP 2011, and standard Fortran implementations.
    """
    def step(self, solver):
        nx = solver.nxmax
        dx = solver.dx
        dt = solver.dt
        gamma = solver.gamma
        gamma1v = 1.0 / (gamma - 1.0)

        qc = solver.qc
        flux = np.zeros((3, nx + 2))  # [1, nx+1] interfaces

        # Loop over all interfaces (including ghost cells)
        for j in range(1, nx + 2):
            # --- Primitive states at the interface ---
            rL = solver.prtvl[0, j - 1]
            uL = solver.prtvl[1, j - 1]
            pL = solver.prtvl[2, j - 1]
            rR = solver.prtvr[0, j]
            uR = solver.prtvr[1, j]
            pR = solver.prtvr[2, j]

            # --- Sound speed ---
            cL = np.sqrt(gamma * pL / rL)
            cR = np.sqrt(gamma * pR / rR)
            # --- Mean inverse sound speed (Fortran: cbv) ---
            cbv = 2.0 / (cL + cR)

            # --- Mach numbers ---
            xm1 = uL * cbv
            xm2 = uR * cbv

            temp = 0.5 * (uL**2 + uR**2)
            xmh = min(1.0, np.sqrt(temp) * cbv)
            chi = (1.0 - xmh) ** 2

            # --- g, unb, un_p, un_m (Fortran: g, unb, un_p, un_m) ---
            g = -max(min(xm1, 0.0), -1.0) * min(max(xm2, 0.0), 1.0)
            unb = (rL * abs(uL) + rR * abs(uR)) / (rL + rR)
            un_p = (1.0 - g) * abs(unb) + g * abs(uL)
            un_m = (1.0 - g) * abs(unb) + g * abs(uR)

            # --- Mass flux (fm) ---
            fm = 0.5 * (rL * (uL + un_p) + rR * (uR - un_m) - chi * (pR - pL) * cbv)
            fmL = 0.5 * (fm + abs(fm))
            fmR = 0.5 * (fm - abs(fm))

            # --- beta+ (left), beta- (right) as in paper ---
            sw1 = 1.0 if abs(xm1) > 1.0 else 0.0
            sw2 = 1.0 if abs(xm2) > 1.0 else 0.0
            sgn_xm1 = 1.0 if xm1 >= 0 else -1.0
            sgn_xm2 = 1.0 if xm2 >= 0 else -1.0

            betaL = (1.0 - sw1) * 0.25 * (2.0 - xm1) * (xm1 + 1.0) ** 2 + sw1 * 0.5 * (1.0 + sgn_xm1)
            betaR = (1.0 - sw2) * 0.25 * (2.0 + xm2) * (xm2 - 1.0) ** 2 + sw2 * 0.5 * (1.0 - sgn_xm2)

            # --- Pressure flux (FVS type, eqn. 32 in AIAA 2009-136) ---
            pbar = 0.5 * ((pL + pR) + (betaL - betaR) * (pL - pR) + (1.0 - chi) * (betaL + betaR - 1.0) * (pL + pR))

            # --- Specific enthalpy (for energy flux) ---
            HL = cL**2 * gamma1v + 0.5 * uL**2
            HR = cR**2 * gamma1v + 0.5 * uR**2

            # --- Fluxes ---
            flux[0, j] = fmL + fmR
            flux[1, j] = fmL * uL + fmR * uR + pbar
            flux[2, j] = fmL * HL + fmR * HR

        # --- Flux divergence and conserved variable update ---
        solver.dflx[:, 1:nx+1] = (flux[:, 2:nx+2] - flux[:, 1:nx+1]) / dx
        solver.qc[:, 1:nx+1] = solver.qc_old[:, 1:nx+1] - dt * solver.dflx[:, 1:nx+1]

        res = np.max(np.abs(solver.dflx[0, 1:nx+1]))
        return res

class KEPScheme:
    """
    Kinetic-Energy-Preserving central scheme + Jameson shock-capturing viscosity with sensor.
    Reference(KEP): Jameson 2008
    Reference(Shock-capturing): Jameson et al., 1981; Shu 1997; 多数のCFD shock-capturing教科書
    """
    def step(self, solver):
        nx = solver.nxmax
        dx = solver.dx
        dt = solver.dt
        gamma = solver.gamma
        gamma1v = solver.gamma1v

        qc = solver.qc
        flux = np.zeros((3, nx+1))

        # --- 1. Physical flux at cell centers
        rho = qc[0]
        u = qc[1] / rho
        E = qc[2]
        p = (E - 0.5 * rho * u**2) / gamma1v
        F = np.zeros_like(qc)
        F[0] = rho * u
        F[1] = rho * u**2 + p
        F[2] = u * (E + p)

        # --- 2. Central flux at interfaces (skew-symmetric/KEP)
        for j in range(nx+1):
            F_left = F[:, j]
            F_right = F[:, j+1]
            flux[:, j] = 0.5 * (F_left + F_right)

        # --- 3. Compute sensor for artificial viscosity ---
        # Sensor based on density gradient (local shock indicator)
        sensor = np.zeros(nx+2)
        beta = 1.0  # sensor sharpness
        for j in range(2, nx):
            # 1. ショックセンサー: 密度2次差分/1次差分の比
            numerator = np.abs(qc[0, j+1] - 2*qc[0, j] + qc[0, j-1])
            denominator = np.abs(qc[0, j+1] - qc[0, j-1]) + 1e-12
            s = numerator / denominator
            # Optionally raise to some power or scale, betaで調整
            sensor[j] = np.tanh(beta * s)

        # --- 4. Apply artificial viscosity with sensor ---
        eps2_base = 0.4
        eps4_base = 0.02
        Q_visc = np.zeros_like(qc)
        for m in range(3):
            # 2nd-order sensor-weighted
            d2 = qc[m, :-2] - 2 * qc[m, 1:-1] + qc[m, 2:]
            # 4th-order sensor-weighted
            d4 = qc[m, :-4] - 4 * qc[m, 1:-3] + 6 * qc[m, 2:-2] - 4 * qc[m, 3:-1] + qc[m, 4:]
            # センサー値の利用: ショック付近は強め、滑らか領域は弱め
            # 2nd: センサーで直接重み付け
            Q_visc[m, 2:-2] = eps2_base * sensor[2:-2] * d2[1:-1] - eps4_base * d4

        # --- 5. Conservative update ---
        solver.dflx[:, 1:nx+1] = (flux[:, 1:nx+1] - flux[:, 0:nx]) / dx
        solver.qc[:, 1:nx+1] = (
            solver.qc_old[:, 1:nx+1]
            - dt * solver.dflx[:, 1:nx+1]
            + dt / dx * (Q_visc[:, 1:nx+1] - Q_visc[:, 0:nx])
        )

        res = np.max(np.abs(solver.dflx[0, 1:nx+1]))
        return res
    
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


class LHS:
    """
    LHS (Left-Hand Side) class for time integration.
    Provides TVD-RK1 (Euler), TVD-RK2, and TVD-RK3 integrators.
    """

    def tvd_rk1(self, solver, scheme):
        """
        TVD-RK1 = Forward Euler.
        Updates solver.qc in place.
        """
        res = scheme.step(solver)
        return res
        
    def tvd_rk2(self, solver, scheme):
        """
        TVD-RK2 (Heun/modified midpoint method).
        Updates solver.qc in place.
        """
        qc0 = solver.qc.copy()

        # Stage 1
        solver.qc = qc0.copy()
        res1 = scheme.step(solver)
        qc1 = solver.qc.copy()

        # Stage 2
        solver.qc = 0.5 * qc0 + 0.5 * qc1
        res2 = scheme.step(solver)
        # Now solver.qc is at t^{n+1}
        return res2
    
    def tvd_rk3(self, solver, scheme):
        """
        TVD-RK3 (Shu-Osher 3rd order) integrator.
        Updates solver.qc in place.
        """
        qc0 = solver.qc.copy()   # Save initial state

        # Stage 1
        solver.qc = qc0.copy()
        res1 = scheme.step(solver)
        qc1 = solver.qc.copy()

        # Stage 2
        solver.qc = 0.75 * qc0 + 0.25 * qc1
        res2 = scheme.step(solver)
        qc2 = solver.qc.copy()

        # Stage 3
        solver.qc = (1.0/3.0) * qc0 + (2.0/3.0) * qc2
        res3 = scheme.step(solver)
        # Now solver.qc is at t^{n+1}
        return res3
