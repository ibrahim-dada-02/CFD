import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

# Self made CFD simulation (Solving 2D laplace solver over a backwards facing step)
class LaplaceFlow:

    def __init__(self, nx = 200, ny = 100, step_height = 0.3, step_width = 0.4):
        # Step width and height are a fraction of the total width and height
        self.nx = nx
        self.ny = ny
        self.Lx = 2.0
        self.Ly = 1.0

        # Create grid
        self.x = np.linspace(0, self.Lx, nx) # Create the x axis
        self.y = np.linspace(0, self.Ly, ny) # Create the y axis
        self.X, self.Y = np.meshgrid(self.x, self.y) # Create the mesh


        # Add the step
        self.step_height = step_height
        self.step_x = int(step_width * nx) # To get the actual coordinates
        self.step_y = int(step_height * ny) # Basically to remove these from the matrix

        # Perform initialization (Assume Nothing)
        self.phi = np.zeros((ny,nx)) # Creates a zero matrix (rws x clmns)

        # Mask the solid region
        self.mask = np.ones((ny,nx), dtype=bool) # Initially set it all to fluid
        self.mask[:self.step_y, :self.step_x] = False # From 0 to y and from 0 to x is set to be a solid

    
    def set_boundary_conditions(self, u_inlet = 1.0): # Initialize boundary condition
        # Inlet
        self.phi[: , 0] = u_inlet * np.linspace(0, self.Ly, self.ny) # The initial velocity is set here

        # Outlet
        self.phi[:, -1] = self.phi[:, -2] # Set it to the same as the interior

    def solve_laplace_sor(self, max_iter = 5000, tol = 1e-5, omega = 1.8):
        # Now we will try to numerically solve this, currently its steady state
        dx = self.Lx / (self.nx - 1) # Discretization is 
        dy = self.Ly / (self.ny - 1) # done here
        dx2 = dx * dx
        dy2 = dy * dy

        # Laplace Equations coeff
        coeff = 1 / (2.0 *(dx2 + dy2))

        # Run solution
        for iteration in range(max_iter):
            phi_old = self.phi.copy()

            # SOR Iteration
            for i in range (1, self.ny - 1): # Solve all y's
                for j in range(1, self.nx - 1): # Solve for all x's
                    if self.mask[i,j]: # Consider only the fluid region
                        # 5-point stencil for Laplace equation
                        phi_new = coeff * (
                            dy2 * (self.phi[i, j+1] + self.phi[i, j-1]) +
                            dx2 * (self.phi[i+1, j] + self.phi[i-1, j])
                        )

                        # SOR update
                        self.phi[i, j] = omega * phi_new + (1 - omega) * self.phi[i, j]
            
            # Apply boundary conditions
            # Top Wall (Zero Gradient for now)
            self.phi[-1, :] = self.phi[-2, :]

            # Bottom wall and step surface (Zero Gradient)
            self.phi[0, self.step_x:] = self.phi[1, self.step_x:]

            # Steps vertical face (Zero Gradient)
            if self.step_x < self.nx: # Checks for non solid
                self.phi[self.step_y:, self.step_x] = self.phi[self.step_y:, self.step_x + 1]

            # Steps horizontal face (Zero Gradient)
            if self.step_y > 0:
                self.phi[self.step_y, :self.step_x] = self.phi[self.step_y+1, :self.step_x]
              
            # Outlet (zero gradient)
            self.phi[:, -1] = self.phi[:, -2]
            
            # Check convergence
            error = np.max(np.abs(self.phi - phi_old))
            
            if iteration % 500 == 0:
                print(f"Iteration {iteration}, Error: {error:.6e}")
            
            if error < tol:
                print(f"Converged in {iteration} iterations")
                break

        return self.phi
    
    def compute_velocity(self):
        """
        Compute velocity field from potential: u = ∂φ/∂x, v = ∂φ/∂y
        """
        dx = self.Lx / (self.nx - 1)
        dy = self.Ly / (self.ny - 1)
        
        # Central differences for interior points
        self.u = np.zeros_like(self.phi)
        self.v = np.zeros_like(self.phi)
        
        # u-velocity
        self.u[:, 1:-1] = (self.phi[:, 2:] - self.phi[:, :-2]) / (2 * dx)
        self.u[:, 0] = (self.phi[:, 1] - self.phi[:, 0]) / dx
        self.u[:, -1] = (self.phi[:, -1] - self.phi[:, -2]) / dx
        
        # v-velocity
        self.v[1:-1, :] = (self.phi[2:, :] - self.phi[:-2, :]) / (2 * dy)
        self.v[0, :] = (self.phi[1, :] - self.phi[0, :]) / dy
        self.v[-1, :] = (self.phi[-1, :] - self.phi[-2, :]) / dy
        
        # Set velocity to zero in solid region
        self.u[~self.mask] = 0
        self.v[~self.mask] = 0
        
        return self.u, self.v
    
    def plot_results(self):
        u, v = self.compute_velocity()
        velocity_mag = np.sqrt(u**2 + v**2)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        contour = ax.contourf(self.X, self.Y, velocity_mag, levels=50, cmap='jet')
        plt.colorbar(contour, ax=ax, label='Velocity Magnitude')
        
        # Add step
        step = Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y], 
                         facecolor='gray', edgecolor='black', linewidth=2)
        ax.add_patch(step)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Flow Over Backward-Facing Step: Velocity Magnitude')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":    
    # Create solver
    solver = LaplaceFlow(nx=200, ny=100, step_height=0.3, step_width=0.2)
    
    # Set boundary conditions
    solver.set_boundary_conditions(u_inlet=1.0)
    
    # Solve Laplace equation
    solver.solve_laplace_sor(max_iter=5000, tol=1e-5, omega=1.8)
    
    # Plot results
    solver.plot_results()

        





