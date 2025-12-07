import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class DNS:
    def __init__(self, nx = 150, ny = 75, step_height = 0.3, step_width = 0.4, Re = 100):
        # Domain
        self.nx = nx
        self.ny = ny
        self.Lx = 2
        self.Ly = 1

        # Grid Size
        self.dx = self.Lx/(nx)
        self.dy = self.Ly/(ny)

        # Create a grid ( Consider the centers rather than the edges )
        self.x = np.linspace(self.dx/2, self.Lx - self.dx / 2, nx)
        self.y = np.linspace(self.dy/2, self.Ly - self.dy / 2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Initialize the parameters
        self.Re = Re
        self.u_inlet = 1.0
        self.nu = self.u_inlet * self.Ly / self.Re
        self.rho = 1.25 # Density of air

        # Initialize the fields
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.u_star = np.zeros((ny, nx))
        self.v_star = np.zeros((ny, nx))

        # Create the step - Mask solids and fluids
        self.step_x = int(step_width * nx)
        self.step_y = int(step_height * ny)

        self.mask = np.ones((ny, nx), dtype=bool)
        self.mask[:self.step_y, :self.step_x] = False

        # Discretise the time
        self.time = 0
        self.total_time = 1

    def initialize_fields(self):
        """Initialize velocities to the inlet"""
        for i in range(self.ny): 
            for j in range(self.nx):
                if self.mask[i, j]: # Loop through all cells except those masked as false
                    self.u[i, j] = 1.0 # Initialise horizontal velocity
                    self.v[i, j] = 0   # Vertical velocity is nill ( Initially )
        
        self.p[:] = 0.0 # Pressure is iniitally constant

    def apply_velocity_bc(self, u, v):
        """Apply all the velocity boundary condition"""
        u[:, 0] = self.u_inlet # all the rows in the first column are the inlet
        v[:, 0] = 0

        u[:, -1] = u[:, -2] # Zero gradient boundary condition
        v[:, -1] = v[:, -2] # Zero gradient boundary condition

        u[-1, :] = 0.0 # No slip condition on the top boundary
        v[-1, :] = 0.0 # No slip condition on the top boundary

        u[0, self.step_x:] = 0.0 # No slip condition on the bottom boundary
        v[0, self.step_y:] = 0.0 # No slip condition on the bottom boundary

        """ An explanation:
        x: means from x till the end
        :x means from 0 till x
        """

        """ Find the boundary of the step since its variable in this code"""
        u[self.step_y, :self.step_x] = 0.0 # Horizontal wall of the step
        v[self.step_y, :self.step_x] = 0.0 # Horizontal wall of the step

        u[:self.step_y, self.step_x] = 0.0 # Vertical wall of the step
        v[:self.step_y, self.step_x] = 0.0 # Vertical wall of the step

        """Applying BC to the solid material"""
        u[~self.mask] = 0.0
        v[~self.mask] = 0.0

    def compute_convection(self, field, u, v): # Call this when we need to solve conv of "v" or "u"
        """Compute the convection term u * del field (Can be any field)"""
        conv = np.zeros_like(field)

        for i in range(1, self.ny - 1): # Only internal cells
            for j in range(1, self.nx - 1):
                if not self.mask[i, j]:
                    continue # Skip if solid

                # Use Central Difference to calculte PDE's
                dfield_dx = (field[i, j + 1] - field[i, j - 1])/(2 * self.dx) # Due to rows x columns its y * x
                dfield_dy = (field[i + 1, j] - field[i - 1, j])/ (2 * self.dy)

                # Calculate convection
                conv[i, j] = u[i , j] * dfield_dx + v[i , j] * dfield_dy
        
        return conv

    def compute_diffusion(self, field):
        """Compute the diffusion laplacian field: del^2 field"""
        laplacian = np.zeros_like(field)

        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if not self.mask[i, j]:
                    continue
                    
                # Hand neighbours using 5 - Point Stencil
                """ For diffusion we require the 5 - Point Stencil because it
                    is a second order differencial equation which require the 
                    gradients from all directions"""
                
                # Check the right one
                if self.mask[i, j + 1]:
                    f_right = field[i, j + 1]
                else:
                    f_right = field[i, j]

                # Check the left one
                if self.mask[i, j - 1]:
                    f_left = field[i, j - 1]
                else:
                    f_left = field[i, j]

                # Check the up one
                if self.mask[i + 1, j]:
                    f_up = field[i + 1, j]
                else:
                    f_up = field[i, j]            
                
                # Check the down one
                if self.mask[i - 1, j]:
                    f_down = field[i - 1, j]
                else:
                    f_down = field[i, j] 

                # 5-point stencil ( Central Difference )
                laplacian[i, j] = (
                        (f_right - 2*field[i, j] + f_left) / self.dx ** 2 +
                        (f_up - 2*field[i, j] + f_down) / self.dy ** 2
                    )
        return laplacian

    def step1_advention_diffusion(self, dt):

        """
        Compute the tempoary velocities
        """
        conv_u = self.compute_convection(self.u, self.u, self.v)
        conv_v = self.compute_convection(self.v, self.u, self.v)

        diff_u = self.compute_diffusion(self.u)
        diff_v = self.compute_diffusion(self.v)

        # Compute diffusion and convection for each cell
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if self.mask[i, j]:
                    self.u_star[i, j] = (self.u[i,j] + dt * (-conv_u[i,j] + self.nu * diff_u[i,j]))
                    
                    self.v_star[i, j] = (self.v[i,j] + dt * (-conv_v[i,j] + self.nu * diff_v[i,j]))
        
        self.apply_velocity_bc(self.u_star, self.v_star)

    def step2_solve_pressure(self, dt, max_iteration = 1000, tol = 1e-4):
        """
        Solve the Possion Equation: del^2 p = rho/dt * del u* (del u = div)
        Incompressibility is also enforced here: del u ^(n + 1) = 0
        """
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        coeff = 1.0 / (2 * (dx2 + dy2))

        # Compute RHS:
        div_u_star = np.zeros_like(self.p) # Initialize div matrix

        for i in range(1, self.ny - 1): # Consider only the internal cells, non boundary
            for j in range(1, self.nx - 1):
                if not self.mask[i,j]:
                    continue
                
                # Use Central Finite to calculate du/dx and dv/dy
                du_dx = (self.u[i, j + 1] - self.u[i, j - 1]) / (2 * self.dx)
                dv_dy = (self.u[i + 1, j] - self.u[i - 1, j]) / (2 * self.dy)

                div_u_star[i, j] = du_dx + dv_dy

        RHS = self.rho * div_u_star / dt

        # Compute LHS: Can be solved using Gauss-Seidal or SOR (Ill use SOR)
        omega = 1.7  # Relaxation parameter (1.0 = Gauss-Seidel, optimal typically 1.5-1.9)

        for iteration in range(max_iteration):
            p_old = self.p.copy()
            
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    if not self.mask[i, j]:
                        continue
                    
                    # Get neighbor pressures
                    p_right = self.p[i, j+1] if self.mask[i, j+1] else self.p[i, j]
                    p_left = self.p[i, j-1] if self.mask[i, j-1] else self.p[i, j]
                    p_up = self.p[i+1, j] if self.mask[i+1, j] else self.p[i, j]
                    p_down = self.p[i-1, j] if self.mask[i-1, j] else self.p[i, j]
                    
                    # Calculate new pressure value
                    p_new = coeff * (
                        dy2 * (p_right + p_left) + 
                        dx2 * (p_up + p_down) - 
                        dx2 * dy2 * RHS[i, j]
                    )
                    
                    # SOR update: blend old and new values
                    self.p[i, j] = (1.0 - omega) * self.p[i, j] + omega * p_new
            
            # Pressure BCs: Neumann (zero gradient at boundaries)
            self.p[:, 0] = self.p[:, 1]
            self.p[:, -1] = self.p[:, -2]
            self.p[0, :] = self.p[1, :]
            self.p[-1, :] = self.p[-2, :]
            
            # Check convergence
            error = np.max(np.abs(self.p[self.mask] - p_old[self.mask]))
            if error < tol:
                break

    def step3_correct_velocity(self, dt):
        # Check every single internal cell (No Boundary)
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if not self.mask[i, j]:
                    continue
                
                # Compute your pressure gradients (Central Difference)
                dp_dx = (self.p[i, j + 1] - self.p[i, j - 1])/ (2 * self.dx)
                dp_dy = (self.p[i + 1, j] - self.p[i - 1, j])/ (2 * self.dy)

                # Correct velocity
                self.u[i, j] = self.u_star[i, j] - dt/self.rho * dp_dx
                self.v[i, j] = self.v_star[i, j] - dt/self.rho * dp_dy

        # Apply velocity boundary conditions
        self.apply_velocity_bc(self.u, self.v)

    def compute_dt_cfl(self, CFL = 0.5): # CFL Should be less than 1
        """Compute time step according to the CFL"""
        u_max = np.max(np.abs(self.u[self.mask]))
        v_max = np.max(np.abs(self.v[self.mask]))

        dt_cfl = CFL * min(self.dx/u_max, self.dy/v_max)

        # Stabalise for diffusion
        dt_diff = 0.25 * min(self.dx ** 2, self.dy ** 2)/self.nu

        """
        Time step should be small enough to prevent, cell skipping and stable diffusion
        """
        return min(dt_cfl, dt_diff)
    
    def time_step(self, dt):
        """Run the code"""
        self.step1_advention_diffusion(dt)
        self.step2_solve_pressure(dt)
        self.step3_correct_velocity(dt)
        self.time += dt

    def solve(self, t_final=5.0, cfl=0.3):
        """Run simulation"""
        print(f"Running to t = {t_final} s")
        
        step_count = 0
        while self.time < t_final:
            dt = self.compute_dt_cfl(cfl)
            self.time_step(dt)
            step_count += 1
            
            if step_count % 100 == 0:
                # Check divergence
                div = np.zeros_like(self.u)
                for i in range(1, self.ny-1):
                    for j in range(1, self.nx-1):
                        if self.mask[i, j]:
                            du_dx = (self.u[i,j+1] - self.u[i,j-1])/(2*self.dx)
                            dv_dy = (self.v[i+1,j] - self.v[i-1,j])/(2*self.dy)
                            div[i, j] = du_dx + dv_dy
                
                max_div = np.max(np.abs(div[self.mask]))
                print(f"  Step {step_count:5d}, t={self.time:.4f}, dt={dt:.6f}, max|∇·u|={max_div:.2e}")
        
        print(f"\nCompleted {step_count} time steps\n")

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
        
        # Velocity magnitude
        c1 = ax1.contourf(self.X, self.Y, vel_masked, levels=50, cmap='jet')
        plt.colorbar(c1, ax=ax1, label='|u|')
        ax1.add_patch(Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y],
                                facecolor='gray', edgecolor='black', linewidth=2))
        ax1.set_title('Velocity Magnitude')
        ax1.set_xlabel('x'); ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        
        # Pressure
        c2 = ax2.contourf(self.X, self.Y, p_masked, levels=50, cmap='RdBu_r')
        plt.colorbar(c2, ax=ax2, label='p')
        ax2.add_patch(Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y],
                                facecolor='gray', edgecolor='black', linewidth=2))
        ax2.set_title('Pressure')
        ax2.set_xlabel('x'); ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        # Vorticity
        c3 = ax3.contourf(self.X, self.Y, vort_masked, levels=50, cmap='RdBu_r')
        plt.colorbar(c3, ax=ax3, label='ω')
        ax3.add_patch(Rectangle((0, 0), self.x[self.step_x], self.y[self.step_y],
                                facecolor='gray', edgecolor='black', linewidth=2))
        ax3.set_title('Vorticity')
        ax3.set_xlabel('x'); ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        
        # Streamlines
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


# Run simulation
if __name__ == "__main__":
    solver = DNS(nx=150, ny=75, step_height=0.3, step_width=0.4, Re=100)
    solver.initialize_fields()
    solver.solve(t_final=5.0, cfl=0.3)
    solver.plot_results()

        

            






