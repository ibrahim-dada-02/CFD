import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import pickle
from numba import njit, prange

class NavierStokesSolver:
    """
    High-performance incompressible Navier-Stokes solver using Chorin's projection method
    Accelerated with Numba JIT compilation and parallel execution
    
    COMPLETE EQUATIONS:
    ∂u/∂t + u·∇u = -∇p + ν∇²u    (momentum)
    ∇·u = 0                        (incompressibility)
    
    Method: Fractional Step (Projection) with Numba parallelization
    """
    
    def __init__(self, nx=500, ny=100, step_height=0.3, step_width=0.1, Re=100):
        """Initialize solver"""
        
        # Domain
        self.nx = nx
        self.ny = ny
        self.Lx = 5.0
        self.Ly = 1.0
        
        self.dx = self.Lx / nx
        self.dy = self.Ly / ny
        
        # Grid (cell centers)
        self.x = np.linspace(self.dx/2, self.Lx - self.dx/2, nx)
        self.y = np.linspace(self.dy/2, self.Ly - self.dy/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Physical parameters
        self.Re = Re
        self.u_inlet = 1.0
        self.nu = self.u_inlet * self.Ly / Re
        self.rho = 1.0
        
        # Fields
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.u_star = np.zeros((ny, nx))
        self.v_star = np.zeros((ny, nx))
        
        # Step geometry
        self.step_x = int(step_width * nx) 
        self.step_y = int(step_height * ny)
        
        # Mask: True = fluid, False = solid
        self.mask = np.ones((ny, nx), dtype=np.bool_)
        self.mask[:self.step_y, :self.step_x] = False
        
        self.time = 0.0
        
        print("="*70)
        print("NUMBA-ACCELERATED NAVIER-STOKES SOLVER")
        print("="*70)
        print(f"Domain: {self.Lx} x {self.Ly} m")
        print(f"Grid: {nx} x {ny} cells")
        print(f"Reynolds number: {Re}")
        print(f"Kinematic viscosity: {self.nu:.6f} m²/s")
        print("Acceleration: Numba JIT + parallel execution")
        print("\nEquations solved:")
        print("  ∂u/∂t + u·∇u = -∇p + ν∇²u")
        print("  ∂v/∂t + u·∇v = -∇p + ν∇²v")
        print("  ∇·u = 0 (incompressibility)")
        print("="*70 + "\n")
    
    def initialize_fields(self):
        """Initialize velocity field"""
        for i in range(self.ny):
            for j in range(self.nx):
                if self.mask[i, j]:
                    self.u[i, j] = self.u_inlet
                    self.v[i, j] = 0.0
        self.p[:] = 0.0
    
    @staticmethod
    @njit
    def _apply_velocity_bc_kernel(u, v, mask, u_inlet, step_x, step_y, ny, nx):
        """Apply velocity boundary conditions (optimized with Numba)"""
        # Inlet
        u[:, 0] = u_inlet
        v[:, 0] = 0.0
        
        # Outlet
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        
        # Top wall
        u[-1, :] = 0.0
        v[-1, :] = 0.0
        
        # Bottom wall
        u[0, step_x:] = 0.0
        v[0, step_x:] = 0.0
        
        # Step top face
        if step_y < ny:
            u[step_y, :step_x] = 0.0
            v[step_y, :step_x] = 0.0
        
        # Step vertical face
        if step_x < nx:
            u[:step_y, step_x] = 0.0
            v[:step_y, step_x] = 0.0
        
        # Zero in solid
        for i in range(ny):
            for j in range(nx):
                if not mask[i, j]:
                    u[i, j] = 0.0
                    v[i, j] = 0.0
    
    def apply_velocity_bc(self, u, v):
        """Apply velocity boundary conditions"""
        self._apply_velocity_bc_kernel(u, v, self.mask, self.u_inlet, 
                                       self.step_x, self.step_y, self.ny, self.nx)
    
    @staticmethod
    @njit(parallel=True)
    def _compute_convection_kernel(field, u, v, mask, dx, dy, ny, nx):
        """Compute convection term: u·∇field (parallelized)"""
        conv = np.zeros_like(field)
        
        for i in prange(1, ny-1):
            for j in range(1, nx-1):
                if not mask[i, j]:
                    continue
                
                dfield_dx = (field[i, j+1] - field[i, j-1]) / (2*dx)
                dfield_dy = (field[i+1, j] - field[i-1, j]) / (2*dy)
                
                conv[i, j] = u[i, j] * dfield_dx + v[i, j] * dfield_dy
        
        return conv
    
    @staticmethod
    @njit(parallel=True)
    def _compute_diffusion_kernel(field, mask, dx, dy, ny, nx):
        """Compute diffusion term: ∇²field (parallelized)"""
        laplacian = np.zeros_like(field)
        
        dx2 = dx**2
        dy2 = dy**2
        
        for i in prange(1, ny-1):
            for j in range(1, nx-1):
                if not mask[i, j]:
                    continue
                
                # Handle neighbors at solid boundaries
                f_right = field[i, j+1] if mask[i, j+1] else field[i, j]
                f_left = field[i, j-1] if mask[i, j-1] else field[i, j]
                f_up = field[i+1, j] if mask[i+1, j] else field[i, j]
                f_down = field[i-1, j] if mask[i-1, j] else field[i, j]
                
                laplacian[i, j] = (
                    (f_right - 2*field[i, j] + f_left) / dx2 +
                    (f_up - 2*field[i, j] + f_down) / dy2
                )
        
        return laplacian
    
    @staticmethod
    @njit(parallel=True)
    def _advection_diffusion_update(u, v, u_star, v_star, conv_u, conv_v, 
                                    diff_u, diff_v, mask, dt, nu, ny, nx):
        """Update intermediate velocity (parallelized)"""
        for i in prange(1, ny-1):
            for j in range(1, nx-1):
                if mask[i, j]:
                    u_star[i, j] = u[i, j] + dt * (-conv_u[i, j] + nu * diff_u[i, j])
                    v_star[i, j] = v[i, j] + dt * (-conv_v[i, j] + nu * diff_v[i, j])
    
    def step1_advection_diffusion(self, dt):
        """STEP 1: Predictor step (no pressure) - NUMBA ACCELERATED"""
        
        # Compute terms using Numba kernels
        conv_u = self._compute_convection_kernel(self.u, self.u, self.v, self.mask, 
                                                 self.dx, self.dy, self.ny, self.nx)
        conv_v = self._compute_convection_kernel(self.v, self.u, self.v, self.mask,
                                                 self.dx, self.dy, self.ny, self.nx)
        
        diff_u = self._compute_diffusion_kernel(self.u, self.mask, self.dx, self.dy, 
                                                self.ny, self.nx)
        diff_v = self._compute_diffusion_kernel(self.v, self.mask, self.dx, self.dy,
                                                self.ny, self.nx)
        
        # Update velocities
        self._advection_diffusion_update(self.u, self.v, self.u_star, self.v_star,
                                        conv_u, conv_v, diff_u, diff_v, 
                                        self.mask, dt, self.nu, self.ny, self.nx)
        
        self.apply_velocity_bc(self.u_star, self.v_star)
    
    @staticmethod
    @njit(parallel=True)
    def _compute_divergence(u_star, v_star, mask, dx, dy, ny, nx):
        """Compute divergence: ∇·u* (parallelized)"""
        div = np.zeros((ny, nx))
        
        for i in prange(1, ny-1):
            for j in range(1, nx-1):
                if not mask[i, j]:
                    continue
                
                du_dx = (u_star[i, j+1] - u_star[i, j-1]) / (2*dx)
                dv_dy = (v_star[i+1, j] - v_star[i-1, j]) / (2*dy)
                
                div[i, j] = du_dx + dv_dy
        
        return div
    
    @staticmethod
    @njit
    def _gauss_seidel_iteration(p, rhs, mask, dx, dy, ny, nx):
        """Single Gauss-Seidel iteration for pressure"""
        dx2 = dx**2
        dy2 = dy**2
        coeff = 1.0 / (2.0 * (dx2 + dy2))
        
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if not mask[i, j]:
                    continue
                
                p_right = p[i, j+1] if mask[i, j+1] else p[i, j]
                p_left = p[i, j-1] if mask[i, j-1] else p[i, j]
                p_up = p[i+1, j] if mask[i+1, j] else p[i, j]
                p_down = p[i-1, j] if mask[i-1, j] else p[i, j]
                
                p[i, j] = coeff * (
                    dy2 * (p_right + p_left) + 
                    dx2 * (p_up + p_down) - 
                    dx2 * dy2 * rhs[i, j]
                )
        
        # Pressure BCs
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
    
    def step2_solve_pressure(self, dt, max_iter=1000, tol=1e-4):
        """STEP 2: Solve pressure Poisson equation - NUMBA ACCELERATED"""
        
        # Compute RHS: ∇·u*
        div_u_star = self._compute_divergence(self.u_star, self.v_star, self.mask,
                                              self.dx, self.dy, self.ny, self.nx)
        rhs = self.rho * div_u_star / dt
        
        # Solve using Gauss-Seidel (JIT compiled)
        for iteration in range(max_iter):
            p_old = self.p.copy()
            
            self._gauss_seidel_iteration(self.p, rhs, self.mask, 
                                        self.dx, self.dy, self.ny, self.nx)
            
            # Check convergence
            error = np.max(np.abs(self.p[self.mask] - p_old[self.mask]))
            if error < tol:
                break
    
    @staticmethod
    @njit(parallel=True)
    def _correct_velocity_kernel(u, v, u_star, v_star, p, mask, dt, rho, dx, dy, ny, nx):
        """Correct velocity with pressure gradient (parallelized)"""
        for i in prange(1, ny-1):
            for j in range(1, nx-1):
                if not mask[i, j]:
                    continue
                
                dp_dx = (p[i, j+1] - p[i, j-1]) / (2*dx)
                dp_dy = (p[i+1, j] - p[i-1, j]) / (2*dy)
                
                u[i, j] = u_star[i, j] - dt/rho * dp_dx
                v[i, j] = v_star[i, j] - dt/rho * dp_dy
    
    def step3_correct_velocity(self, dt):
        """STEP 3: Correct velocity with pressure gradient - NUMBA ACCELERATED"""
        self._correct_velocity_kernel(self.u, self.v, self.u_star, self.v_star,
                                     self.p, self.mask, dt, self.rho,
                                     self.dx, self.dy, self.ny, self.nx)
        self.apply_velocity_bc(self.u, self.v)
    
    def compute_dt_cfl(self, cfl=0.5):
        """Compute time step from CFL condition"""
        u_max = np.max(np.abs(self.u[self.mask])) + 1e-10
        v_max = np.max(np.abs(self.v[self.mask])) + 1e-10
        
        dt_cfl = cfl * min(self.dx / u_max, self.dy / v_max)
        dt_diff = 0.25 * min(self.dx**2, self.dy**2) / self.nu
        
        return min(dt_cfl, dt_diff)
    
    def time_step(self, dt):
        """Execute one full time step of projection method"""
        self.step1_advection_diffusion(dt)
        self.step2_solve_pressure(dt)
        self.step3_correct_velocity(dt)
        self.time += dt
    
    def save_snapshot(self, folder='data', frame_num=0):
        """Save current state to file"""
        os.makedirs(folder, exist_ok=True)
        
        data = {
            'time': self.time,
            'u': self.u.copy(),
            'v': self.v.copy(),
            'p': self.p.copy(),
            'X': self.X,
            'Y': self.Y,
            'mask': self.mask,
            'step_x': self.step_x,
            'step_y': self.step_y,
            'Lx': self.Lx,
            'Ly': self.Ly,
            'x': self.x,
            'y': self.y,
            'dx': self.dx,
            'dy': self.dy
        }
        
        filename = os.path.join(folder, f'snapshot_{frame_num:04d}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved snapshot at t={self.time:.3f}s to {filename}")
    
    def solve(self, t_final=1.0, save_interval=0.1, cfl=0.3):
        """Run simulation with periodic data saving"""
        print(f"Running to t = {t_final} s")
        print(f"Saving data every {save_interval} s")
        print("First iteration will be slow (Numba compilation)...\n")
        
        step_count = 0
        frame_num = 0
        time_since_save = 0.0
        
        self.save_snapshot(frame_num=frame_num)
        frame_num += 1
        
        while self.time < t_final:
            dt = self.compute_dt_cfl(cfl)
            self.time_step(dt)
            step_count += 1
            time_since_save += dt
            
            if time_since_save >= save_interval:
                self.save_snapshot(frame_num=frame_num)
                frame_num += 1
                time_since_save = 0.0
            
            if step_count % 100 == 0:
                # Check divergence
                div = self._compute_divergence(self.u, self.v, self.mask,
                                              self.dx, self.dy, self.ny, self.nx)
                max_div = np.max(np.abs(div[self.mask]))
                print(f"  Step {step_count:5d}, t={self.time:.4f}, dt={dt:.6f}, max|∇·u|={max_div:.2e}")
        
        if time_since_save > 0:
            self.save_snapshot(frame_num=frame_num)
        
        print(f"\nCompleted {step_count} time steps")
        print(f"Saved {frame_num + 1} snapshots\n")
    
    def plot_results(self):
        """Visualize solution"""
        
        # Compute vorticity
        vorticity = np.zeros_like(self.u)
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                if self.mask[i, j]:
                    dv_dx = (self.v[i, j+1] - self.v[i, j-1]) / (2*self.dx)
                    du_dy = (self.u[i+1, j] - self.u[i-1, j]) / (2*self.dy)
                    vorticity[i, j] = dv_dx - du_dy
        
        velocity_mag = np.sqrt(self.u**2 + self.v**2)
        vel_masked = np.ma.array(velocity_mag, mask=~self.mask)
        vort_masked = np.ma.array(vorticity, mask=~self.mask)
        p_masked = np.ma.array(self.p, mask=~self.mask)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        c1 = ax1.contourf(self.X, self.Y, vel_masked, levels=50, cmap='jet')
        plt.colorbar(c1, ax=ax1, label='|u|')
        ax1.add_patch(Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y],
                                facecolor='gray', edgecolor='black', linewidth=2))
        ax1.set_title('Velocity Magnitude')
        ax1.set_xlabel('x'); ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        
        c2 = ax2.contourf(self.X, self.Y, p_masked, levels=50, cmap='RdBu_r')
        plt.colorbar(c2, ax=ax2, label='p')
        ax2.add_patch(Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y],
                                facecolor='gray', edgecolor='black', linewidth=2))
        ax2.set_title('Pressure')
        ax2.set_xlabel('x'); ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        c3 = ax3.contourf(self.X, self.Y, vort_masked, levels=50, cmap='RdBu_r')
        plt.colorbar(c3, ax=ax3, label='ω')
        ax3.add_patch(Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y],
                                facecolor='gray', edgecolor='black', linewidth=2))
        ax3.set_title('Vorticity')
        ax3.set_xlabel('x'); ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        
        ax4.streamplot(self.X, self.Y, self.u, self.v, 
                      color=velocity_mag, cmap='jet', density=2)
        ax4.add_patch(Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y],
                                facecolor='gray', edgecolor='black', linewidth=2))
        ax4.set_title('Streamlines')
        ax4.set_xlabel('x'); ax4.set_ylabel('y')
        ax4.set_aspect('equal')
        ax4.set_xlim(0, self.Lx); ax4.set_ylim(0, self.Ly)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    solver = NavierStokesSolver(nx=150, ny=75, step_height=0.3, step_width=0.1, Re=100)
    solver.initialize_fields()
    solver.solve(t_final=5.0, save_interval=0.1, cfl=0.3)
    solver.plot_results()
    
    print("✅ Numba-accelerated Navier-Stokes simulation complete!")
    print("\nPhysics captured:")
    print("  ✓ Convection (u·∇u) - NUMBA PARALLEL")
    print("  ✓ Diffusion (ν∇²u) - NUMBA PARALLEL")
    print("  ✓ Pressure (∇p) - NUMBA JIT")
    print("  ✓ Incompressibility (∇·u = 0)")
    print("  ✓ Velocity correction - NUMBA PARALLEL")
    print("\nData saved in 'data' folder")