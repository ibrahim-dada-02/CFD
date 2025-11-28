import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Scratch:
    def __init__(self, nx=200, ny=100, step_height=0.3, step_width=0.4):
        """Initialize the computational domain"""
        
        # Domain parameters
        self.nx = nx  # Number of cells horizontally
        self.ny = ny  # Number of cells vertically
        self.Lx = 2.0  # Length of domain (meters)
        self.Ly = 1.0  # Height of domain (meters)
        
        # Create spatial grid
        # FIX 1: Should go from 0 to Lx/Ly, not 0 to nx/ny!
        self.x = np.linspace(0, self.Lx, nx)  # ✓ Fixed
        self.y = np.linspace(0, self.Ly, ny)  # ✓ Fixed
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize potential field
        self.phi = np.zeros((ny, nx))
        
        # Create step geometry
        self.step_x = int(step_width * nx)
        self.step_y = int(step_height * ny)
        
        # Create mask: True = fluid, False = solid
        self.mask = np.ones((ny, nx), dtype=bool)
        self.mask[:self.step_y, :self.step_x] = False
        
    def initialise_fields(self, u_inlet=1.0):
        """Initialize the potential field with uniform flow guess"""
        self.u_inlet = u_inlet
        
        # Initialize with linear potential (good initial guess)
        for i in range(self.ny):
            for j in range(self.nx):
                if self.mask[i, j]:
                    self.phi[i, j] = u_inlet * self.x[j]
    
    def set_boundary_conditions(self):
        """Apply all boundary conditions"""
        
        dx = self.Lx / (self.nx - 1)
        
        # 1. Inlet (Dirichlet): fixed value
        self.phi[:, 0] = 0
        
        # 2. Outlet (Neumann): fixed gradient
        self.phi[:, -1] = self.phi[:, -2] + self.u_inlet * dx
        
        # 3. Top wall (Neumann): zero gradient
        self.phi[-1, :] = self.phi[-2, :]
        
        # 4. Bottom wall (Neumann): zero gradient
        self.phi[0, self.step_x:] = self.phi[1, self.step_x:]
        
        # 5. Step top face (Neumann): zero gradient
        if self.step_y < self.ny - 1:
            self.phi[self.step_y, :self.step_x] = self.phi[self.step_y + 1, :self.step_x]
        
        # 6. Step vertical face (Neumann): zero gradient
        if self.step_x < self.nx - 1:
            for i in range(self.step_y):
                self.phi[i, self.step_x] = self.phi[i, self.step_x + 1]
    
    def solve_laplace_sor(self, max_iter=5000, tol=1e-5, omega=1.8):
        """Solve Laplace equation using SOR method"""
        
        dx = self.Lx / (self.nx - 1)
        dy = self.Ly / (self.ny - 1)
        dx2 = dx * dx
        dy2 = dy * dy
        coeff = 1.0 / (2.0 * (dx2 + dy2))
        
        print("Starting SOR iteration...")
        
        for iteration in range(max_iter):
            phi_old = self.phi.copy()
            
            # SOR sweep over all interior points
            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    # Skip solid cells
                    if not self.mask[i, j]:
                        continue
                    
                    # Skip cells adjacent to inlet (Dirichlet BC)
                    if j == 1:
                        continue
                    
                    # FIX 2: Check correct indices for each neighbor!
                    
                    # Right neighbor (j+1)
                    if self.mask[i, j+1]:  # ✓ Correct
                        phi_right = self.phi[i, j+1]
                    else:
                        phi_right = self.phi[i, j]
                    
                    # Left neighbor (j-1)
                    if self.mask[i, j-1]:  # ✓ Fixed: was j+1
                        phi_left = self.phi[i, j-1]
                    else:
                        phi_left = self.phi[i, j]
                    
                    # Up neighbor (i+1)
                    if self.mask[i+1, j]:  # ✓ Fixed: was [i, j+1]
                        phi_up = self.phi[i+1, j]
                    else:
                        phi_up = self.phi[i, j]
                    
                    # Down neighbor (i-1)
                    if self.mask[i-1, j]:  # ✓ Fixed: was [i, j+1]
                        phi_down = self.phi[i-1, j]
                    else:
                        phi_down = self.phi[i, j]
                    
                    # Compute new value using 5-point stencil
                    phi_new = coeff * (
                        dy2 * (phi_right + phi_left) + 
                        dx2 * (phi_up + phi_down)
                    )
                    
                    # SOR update
                    self.phi[i, j] = omega * phi_new + (1 - omega) * self.phi[i, j]
            
            # FIX 3: Apply BCs INSIDE the loop, after each iteration!
            self.set_boundary_conditions()
            
            # Check convergence
            error = np.max(np.abs(self.phi[self.mask] - phi_old[self.mask]))
            
            if iteration % 500 == 0:
                print(f"Iteration {iteration:5d}, Error: {error:.6e}")
            
            if error < tol:
                print(f"\nConverged in {iteration} iterations with error {error:.6e}")
                break
        else:
            print(f"\nWarning: Max iterations reached. Error: {error:.6e}")
        
        return self.phi
    
    def compute_velocity(self):
        """Compute velocity field from potential"""
        dx = self.Lx / (self.nx - 1)
        dy = self.Ly / (self.ny - 1)
        
        self.u = np.zeros_like(self.phi)
        self.v = np.zeros_like(self.phi)
        
        # u = ∂φ/∂x
        self.u[:, 1:-1] = (self.phi[:, 2:] - self.phi[:, :-2]) / (2 * dx)
        self.u[:, 0] = (self.phi[:, 1] - self.phi[:, 0]) / dx
        self.u[:, -1] = (self.phi[:, -1] - self.phi[:, -2]) / dx
        
        # v = ∂φ/∂y
        self.v[1:-1, :] = (self.phi[2:, :] - self.phi[:-2, :]) / (2 * dy)
        self.v[0, :] = (self.phi[1, :] - self.phi[0, :]) / dy
        self.v[-1, :] = (self.phi[-1, :] - self.phi[-2, :]) / dy
        
        # Zero velocity in solid
        self.u[~self.mask] = 0
        self.v[~self.mask] = 0
        
        return self.u, self.v
    
    def plot_results(self):
        """Visualize the solution"""
        u, v = self.compute_velocity()
        velocity_mag = np.sqrt(u**2 + v**2)
        velocity_mag_masked = np.ma.array(velocity_mag, mask=~self.mask)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        # Velocity magnitude
        contour = ax1.contourf(self.X, self.Y, velocity_mag_masked, 
                               levels=50, cmap='jet')
        plt.colorbar(contour, ax=ax1, label='Velocity Magnitude')
        
        step = Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y],
                        facecolor='gray', edgecolor='black', linewidth=2)
        ax1.add_patch(step)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Velocity Magnitude')
        ax1.set_aspect('equal')
        
        # Streamlines
        speed = np.sqrt(u**2 + v**2)
        ax2.streamplot(self.X, self.Y, u, v, color=speed, 
                      cmap='jet', density=2, linewidth=1)
        
        step2 = Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y],
                         facecolor='gray', edgecolor='black', linewidth=2)
        ax2.add_patch(step2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Streamlines')
        ax2.set_aspect('equal')
        ax2.set_xlim(0, self.Lx)
        ax2.set_ylim(0, self.Ly)
        
        plt.tight_layout()
        plt.show()


# Test run
if __name__ == "__main__":
    solver = Scratch(nx=200, ny=100, step_height=0.3, step_width=0.4)
    solver.initialise_fields(u_inlet=1.0)
    solver.set_boundary_conditions()
    solver.solve_laplace_sor(max_iter=5000, tol=1e-5, omega=1.8)
    solver.plot_results()
    
    print("\n✅ Simulation complete!")
    print(f"Phi range: [{solver.phi.min():.3f}, {solver.phi.max():.3f}]")
