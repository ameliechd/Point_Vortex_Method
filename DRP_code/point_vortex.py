import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from scipy.special import j0, j1

############################################################################### First Point Vortex code!! 
class PointVortex: 

    def __init__(self, r,  Gamma, dt):
        ''' Initializing variables'''
        self.r = r.astype(float) #matrix with x and y postions as columns [[x1,...,xN],[y1,...,yN]]
        self.Gamma = Gamma #Circulation 
        self.dt = dt #time step 
        self.timeElapsed = 0

        #for circle boundary
        self.R = 3.0 #circle radius
        self.center = np.array([0.0,0.0]) #center of circle

    def vortex_velocity(self, r):
        ''' Compute vortex velocity using point vortex method, but vectorized style
        using 3 matrices. The first two are matrices that calculate the differences between 
        the x's and y's (ex: x_i - x_j where i!=j). The other matrix is for the denominator
        1/r**2. The full calculation for velocities is then done via Gamma vector and matrix 
        multiplication.'''

        rT = np.transpose(r)
        n = len(rT)

        #x matrices 
        x_matrix_T = np.tile(r[0], (n, 1))
        x_matrix = x_matrix_T.T
        x_diff = x_matrix - x_matrix_T

        #y matrices
        y_matrix_T = np.tile(r[1], (n, 1))
        y_matrix = y_matrix_T.T
        y_diff = y_matrix - y_matrix_T

        #r norm squared matrices 
        r_squared = x_diff*x_diff + y_diff*y_diff # r squared matrix 
        r_recip = np.reciprocal(r_squared, where=(r_squared != 0))  #reciprocal
        r_recip = np.nan_to_num(r_recip, nan=0) #making the nans 0 

        #matrix multiplication (summation of the vortex contributions)
        vx = -(1/(2*np.pi))*((y_diff*r_recip)@self.Gamma)
        vy = (1/(2*np.pi))*((x_diff*r_recip)@self.Gamma)    

        return vx, vy
    
    #circle boundary style
    def vortex_velocity_circle(self, r=None):
        ''' Similar to the point vortex method but now with added cicular boundary. Solved with
        by adding image vortices (via circle theorem).'''

        if r is None:
            r = self.r

        #real velocities 
        vx, vy = self.vortex_velocity(r)
        
        #image vortices 
        x, y = r[0], r[1]
        r_squared = (x)**2 + (y)**2 #norm squared
        x_im = self.R**2 / r_squared * x 
        y_im = self.R**2 / r_squared * y

        r_im = np.array([x_im, y_im])
        Gamma_im = -self.Gamma #inv circulation

        #velocity of images 
        rT_im = np.transpose(r_im)
        n=len(rT_im)

        x_matrix_T_im = np.tile(r_im[0], (n,1))
        x_matrix_im = np.transpose(x_matrix_T_im)
        x_matrix_T = np.tile(r[0], (n, 1)) #normal guys!
        x_matrix = x_matrix_T.T
        x_diff_im = x_matrix - x_matrix_T_im

        y_matrix_T_im = np.tile(r_im[1], (n,1))
        y_matrix_im = np.transpose(y_matrix_T_im)
        y_matrix_T = np.tile(r[1], (n, 1))
        y_matrix = y_matrix_T.T
        y_diff_im = y_matrix - y_matrix_T_im

        r_squared_im = x_diff_im**2 + y_diff_im**2 +(1e-3)**2
        r_recip_im = np.reciprocal(r_squared_im, where=(r_squared_im != 0))
        r_recip_im = np.nan_to_num(r_recip_im, nan=0) 

        vx_im = -(1/(2*np.pi)) * ((y_diff_im * r_recip_im) @ Gamma_im)
        vy_im = (1/(2*np.pi)) * ((x_diff_im * r_recip_im) @ Gamma_im)   

        #total velocity    
        vx_tot = vx + vx_im
        vy_tot = vy + vy_im

        return vx_tot, vy_tot
    

    #methods for numerical integration
    def RK4(self, vx, vy): 
        ''' Updates position using RK4'''

        #doing this dumb style and sperating x, y components 
        x, y = self.r[0], self.r[1] 
        
        #we update coessficients and then we direct;y update position
        
        k1x = vx 
        k1y = vy 

        k2x, k2y = self.vortex_velocity(np.array([k1x*self.dt/2, k1y*self.dt/2])+self.r) #HOW TO UPDATE 

        k3x, k3y = self.vortex_velocity(np.array([k2x*self.dt/2, k2y*self.dt/2])+self.r)

        k4x, k4y = self.vortex_velocity(np.array([k3x*self.dt, k3y*self.dt])+self.r)  

        #position update 
        x = x + (k1x+ 2*k2x + 2*k3x + k4x)*self.dt/6
        y = y + (k1y+ 2*k2y + 2*k3y + k4y)*self.dt/6

        self.r = np.array([x, y]) #putting it back in r 
        #put back on boundary for circle 
        return self.r
    
    
    def RK4_circle(self, vx, vy): #delete this repetition and make cleaner
        ''' Updates position using RK4'''

        #doing this dumb style and sperating x, y components 
        x, y = self.r[0], self.r[1] 
        
        #we update coefficients and then we direct;y update position
        
        k1x = vx 
        k1y = vy 

        k2x, k2y = self.vortex_velocity_circle(np.array([k1x*self.dt/2, k1y*self.dt/2])+self.r) #HOW TO UPDATE 

        k3x, k3y = self.vortex_velocity_circle(np.array([k2x*self.dt/2, k2y*self.dt/2])+self.r)

        k4x, k4y = self.vortex_velocity_circle(np.array([k3x*self.dt, k3y*self.dt])+self.r)  

        #position update 
        x += (k1x+ 2*k2x + 2*k3x + k4x)*self.dt/6
        y += (k1y+ 2*k2y + 2*k3y + k4y)*self.dt/6

        self.r = np.array([x, y]) #putting it back in r 

        return self.r
    
    
    #testing other numerical integration methods 
    def backward_euler(self, vx, vy): #FIX
        ''' uses initial step from forward euler once then backwards euler's the rest'''
        
        #separating 
        x, y = self.r[0], self.r[1]

        #forward step 
        x_f = x + vx*self.dt
        y_f = y + vy*self.dt 

        #backward steps 
        v_xf, v_yf = self.vortex_velocity(np.array([x_f, y_f]))
        x += v_xf*self.dt
        y += v_yf*self.dt

        self.r = np.array([x, y])

        return self.r
    
    def trapezoidal(self, vx, vy):
        ''' Trapezoidal rule with a forward euler step to begin'''

        x, y = self.r[0], self.r[1]

        #forward step 
        x_f = x + vx*self.dt
        y_f = y + vy*self.dt 

        #trapezoidal rule 
        x += (self.dt / 2) * (vx + self.vortex_velocity(np.array([x_f, y_f]))[0])
        y += (self.dt / 2) * (vy + self.vortex_velocity(np.array([x_f, y_f]))[1])

        self.r = np.array([x,y])

        return self.r

    def midpoint(self, vx, vy):
        ''' Midpoint method with a forward euler step to begin'''

        x, y = self.r[0], self.r[1]

        # Half-step position estimate
        x_half = x + vx * (self.dt / 2)
        y_half = y + vy * (self.dt / 2)

        # Compute velocity at the midpoint
        vx_half, vy_half = self.vortex_velocity(np.array([x_half, y_half]))

        # Full-step using midpoint velocity
        x += vx_half * self.dt
        y += vy_half * self.dt

        self.r = np.array([x, y])
        return self.r

    # many plotting function ahead
    def point_animate(self, runs=100, interval=50, skip_frame=1): 
        ''' Animate the point vortex motion'''
        # Setting up the plot
        fig, ax = plt.subplots(figsize=(7, 7))
        #circle = plt.Circle((5, 5), 2, color='black', fill=False, linestyle='-', linewidth=1.5, alpha =.2)
        #ax.add_artist(circle)
        ax.set_xlim(-1, 1) #make this aribitrary?
        ax.set_ylim(-1, 1)
        ax.set_xlabel('x position', fontsize = 15)
        ax.set_ylabel('y position', fontsize = 15)
        # normalize Gamma for color mapping
        #norm = SymLogNorm( 1e-300,vmin=np.min(self.Gamma), vmax=np.max(self.Gamma))
        scat = ax.scatter(self.r[0], self.r[1], linewidth = 0.05, marker='.', c = self.Gamma, cmap ='jet') #creating scatter (norm) linewidth = 0.05, marker='.', c = self.Gamma, cmap ='jet'
        #scat = ax.contourf(self.r[0], self.y[0], self.Gamma, 100, cmap = 'jet')
        fig.savefig("first_frame_circle.png", dpi=300)

        #running the frames
        def update(frame):
            ''' Update function for the animation '''
            for _ in range(skip_frame): #changes how many frames you loop over 
                vx, vy = self.vortex_velocity(self.r)
                self.timeElapsed += self.dt
                self.RK4(vx, vy) #change this if you want to change method used 
            scat.set_offsets(self.r.T)  # Update particle positions  
            ax.set_title(f'Point Vortex Animation, {len(self.Gamma)} points, t={round(self.timeElapsed,2)}', fontsize= 18)
            #print(np.max(np.abs(np.sqrt((self.r[0]-5)**2+(self.r[1]-5)**2)-2))) #printing circle divergence
            return scat,

        # the actual animation
        ani = FuncAnimation(fig, update, frames=runs, interval=interval)

        #cbar = plt.colorbar(scat, ax=ax)
        #cbar.set_label('Gamma Magnitude')

        plt.show()

    def point_animate_circle(self, runs=100, interval=50, skip_frame=1): 
        ''' Animate the point vortex motion but with circular boundary'''
        # Setting up the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        circle = plt.Circle((0, 0), self.R, color='black', fill=False, linestyle='-', linewidth=1.5) #added circle
        ax.add_artist(circle)
        ax.set_xlim(-4, 4) 
        ax.set_ylim(-4, 4)
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        # normalize Gamma for color mapping
        scat = ax.scatter(self.r[0], self.r[1], linewidth = 0.05, marker='.', c = self.Gamma, cmap ='jet') #creating scatter

        #running the frames
        def update(frame):
            ''' Update function for the animation '''
            for _ in range(skip_frame): #changes how many frames you loop over 
                vx, vy = self.vortex_velocity_circle()
                self.timeElapsed += self.dt
                self.RK4_circle(vx, vy) #change this if you want to change method used 
            scat.set_offsets(self.r.T)  # Update particle positions  
            ax.set_title(f'Point Vortex Animation Circular Boundary, {len(self.Gamma)} points, t={round(self.timeElapsed,2)}')
            #print(np.max(np.abs(np.sqrt((self.r[0]-5)**2+(self.r[1]-5)**2)-2))) #printing circle divergence
            return scat,

        # the actual animation
        ani = FuncAnimation(fig, update, frames=runs, interval=interval)

        plt.show()

    #alternative animation with mesh grid
    def mesh_animate(self, runs=100, interval=50, skip_frame =1): ####FIX FIX FIX FIX
        ''' animates but meshgrid contour plot style'''

        #creating grid 
        a = np.linspace(-5, 5, 200)
        #X, Y = np.meshgrid(a, a)

        #setting up the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        contour = ax.tricontourf(self.r[0], self.r[1], self.Gamma, levels=256, cmap='jet') #put back the self.r[i]

        plt.axis("equal")
        plt.tight_layout()
        #plt.show()

        def update(frame):
            nonlocal contour

            #evolving using point vortex method
            for _ in range(skip_frame):
                vx, vy = self.vortex_velocity(self.r)
                self.RK4(vx, vy)  #can swap with midpoint method 
                self.timeElapsed += self.dt

            #recalculate the scalar field Z using Gamma and updated positions (this is slowing down sim)
            Z = self.Gamma #np.zeros_like(X)

            ax.clear()
            contour = ax.tricontourf(self.r[0], self.r[1], self.Gamma, levels=256, cmap='jet')

            plt.title(f't={self.timeElapsed}')

            return []

        # Run the animation
        ani = FuncAnimation(fig, update, frames=runs, interval=1, blit=False)
        plt.show()

    def mesh_snapshot(self, runs=1000, skip_frame=1, num_snapshots=5, save_prefix="snapshot"):
        """
        Run the vortex simulation and save a few contour snapshots (not animated).
        
        Parameters:
        - runs: total number of time steps.
        - skip_frame: steps per update (same meaning as in animation).
        - num_snapshots: number of frames to save.
        - save_prefix: filename prefix for saved plots.
        """

        # Set up snapshot intervals
        snapshot_indices = np.linspace(0, runs, num_snapshots, endpoint=False, dtype=int)
        
        snapshot_counter = 0

        for i in range(runs):
            for _ in range(skip_frame):
                vx, vy = self.vortex_velocity(self.r)  # Or vortex_velocity_circle()
                self.RK4(vx,vy)
                self.timeElapsed += self.dt

            if i in snapshot_indices:
                plt.figure(figsize=(6, 6))
                plt.tricontourf(self.r[0], self.r[1], self.Gamma, levels=256, cmap='jet')
                plt.title(f't = {self.timeElapsed:.5f}')
                plt.xlabel('x position')
                plt.ylabel('y position')
                plt.axis("equal")
                plt.tight_layout()
                filename = f"{save_prefix}_{snapshot_counter:03d}.png"
                plt.savefig(filename, dpi=300)
                plt.close()
                snapshot_counter += 1

    def plot_initial_condition_mesh(self, save_prefix="initial_condition"):
        plt.figure(figsize=(6, 6))
        plt.tricontourf(self.r[0], self.r[1], self.Gamma, levels=256, cmap='jet')
        plt.title('Initial Condition')
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{save_prefix}.png", dpi=300)
        plt.close()

    def scatter_snapshot(self, runs=1000, skip_frame=1, num_snapshots=5, save_prefix="scatter_snapshot"):
        """
        Run the vortex simulation and save scatter plot snapshots (points colored by Gamma).

        Parameters:
        - runs: total number of time steps.
        - skip_frame: steps per update.
        - num_snapshots: number of frames to save.
        - save_prefix: filename prefix for saved plots.
        """
        snapshot_indices = np.linspace(0, runs, num_snapshots, endpoint=False, dtype=int)
        snapshot_counter = 0

        for i in range(runs):
            for _ in range(skip_frame):
                vx, vy = self.vortex_velocity(self.r)
                self.RK4(vx, vy)
                self.timeElapsed += self.dt

            if i in snapshot_indices:
                plt.figure(figsize=(7, 7))
                plt.scatter(self.r[0], self.r[1], c=self.Gamma, cmap='jet', s=10, edgecolor='k', linewidth=0.1)
                plt.colorbar(label='Gamma')
                plt.title(f'Scatter Snapshot t = {self.timeElapsed:.5f}')
                plt.xlabel('x position')
                plt.ylabel('y position')
                plt.axis('equal')
                plt.tight_layout()
                filename = f"{save_prefix}_{snapshot_counter:03d}.png"
                plt.savefig(filename, dpi=300)
                plt.close()
                snapshot_counter += 1

    def plot_initial_condition(self, save_prefix="initial_condition"):
        """
        Plot and save the initial vortex positions as a scatter plot colored by Gamma.
        """
        plt.figure(figsize=(8, 7))
        plt.scatter(self.r[0], self.r[1], c=self.Gamma, cmap='jet', s=10, edgecolor='k', linewidth=0.1)
        plt.colorbar(label='Gamma')
        plt.title('Initial Condition', fontsize =18)
        plt.xlabel('x position', fontsize =15)
        plt.ylabel('y position', fontsize =15)
        plt.axis('equal')
        plt.tight_layout()
        filename = f"{save_prefix}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

    def circle_snapshots(self, runs=1000, skip_frame=1, num_snapshots=5, save_prefix="circle_snapshot"):
        """
        Run the vortex simulation with circular boundary and save snapshots of vortex positions colored by Gamma.
        
        Parameters:
        - runs: total number of time steps
        - skip_frame: steps per update (like in animation)
        - num_snapshots: number of snapshots to save
        - save_prefix: filename prefix for saved images
        """

        snapshot_indices = np.linspace(0, runs, num_snapshots, endpoint=False, dtype=int)
        snapshot_counter = 0

        for i in range(runs):
            for _ in range(skip_frame):
                vx, vy = self.vortex_velocity_circle()
                self.timeElapsed += self.dt
                self.RK4_circle(vx, vy)

            if i in snapshot_indices:
                plt.figure(figsize=(6, 6))
                ax = plt.gca()
                circle = plt.Circle(self.center, self.R, color='black', fill=False, linestyle='-', linewidth=1.5)
                ax.add_artist(circle)
                scatter = ax.scatter(self.r[0], self.r[1], c=self.Gamma, cmap='jet', marker='.', s=15, edgecolors='none')
                #plt.colorbar(scatter, label='Circulation (Gamma)')
                plt.title(f'Circular Boundary Snapshots, t={self.timeElapsed:.5f}')
                plt.xlabel('x position')
                plt.ylabel('y position')
                plt.axis('equal')
                plt.xlim(-3.5,3.5)
                plt.ylim(-3.5,3.5)
                #plt.tight_layout()
                filename = f"{save_prefix}_{snapshot_counter:03d}.png"
                plt.savefig(filename, dpi=300)
                plt.close()
                snapshot_counter += 1

################################################################################# Testing 

''' To run: create your point vortices with PointVortex(initial position array, Gamma array, time step)
    then call a snapsot method on your newly created point vortices as .mesh_snapshot() or .point_snapsot() 
    Make sure to ave good GPU. For reference, running 10 000 pv's on 16GB of RAM seemed to be my limit.'''


'''Vortices in a circle'''
npts = 8
dtheta = 2*np.pi/npts
theta = np.linspace(0, 2*np.pi - dtheta, npts)
xs = 5 + 2*np.cos(theta)
ys = 5 + 2*np.sin(theta)
#pv2 = PointVortex(np.array([xs, ys]), np.array([4.,4.,4.,4.,4.,4.,4.,4.]), 0.005)
#pv2.point_animate(runs = 100, interval = 50)

'''Vortices in a line at x =5'''
#pv3 = PointVortex(np.array([[5,5,5,5,5,5,5,5,5],[1,2,3,4,5,6,7,8,9]]), np.array([3,3,3,3,3,3,3,3,3]), 0.05)
#pv3.animate(runs = 100, interval = 50)

'''test with equilateral triangle '''
#pv4 = PointVortex(np.array([[2,5,2+1.5],[2,2,2+np.sqrt(3**2-(1.5)**2)]]) +1.5, np.array([3,3,3]), 0.05)
#pv4.animate(runs = 100, interval = 50 )

''' Random vortices '''
#test with random points 
#x = np.random.uniform(-3, 3, 1000)
#y = np.random.uniform(-3, 3, 1000)
#gamma = np.random.uniform(-10, 10, 50)

#pv5 = PointVortex(np.array([x,y]), gamma, 0.002)
#pv5.animate(runs = 100, interval = 50)
#small dt, many N

'''Two-vortex initial condition '''
x = np.random.uniform(-0.5, 0.5, 20000)
y = np.random.uniform(-0.5, 0.5, 20000)
#x= np.linspace(-0.5, 0.5, 2000)
#y=np.linspace(-0.5, 0.5, 2000)

def g(x, y): 
    sig = 0.07
    r = 1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*(x**2+y**2)/sig**2)
    return r

omega = -(g(x-0.15, y-0) + g(x+0.15, y+0)) #this is the same as Gamma 
#pv6 = PointVortex(np.array([x,y]), omega, 0.0001) #make point-vortices
#pv6.plot_initial_condition(save_prefix="initial_condition")
#pv6.mesh_snapshot(runs=10, skip_frame=1, num_snapshots=10, save_prefix="two_vortex_frame") #THIS
#pv6.point_animate(runs=100, interval=50, skip_frame=1)
#pv6.plot_initial_condition(save_prefix="point_initial_condition")
#pv6.scatter_snapshot(runs=1000, skip_frame=1, num_snapshots=5, save_prefix="two_vortex_scatter")

'''Circle boundary condition w/ lamb-chaplyin dipole '''

#bessel functions of first kind 
J0 = j0 
J1 = j1

R_dipole = 0.1
k = 3.831705970207515/R_dipole
U = 0.1 #1/3 ?

lamb_dipole = lambda x, y: np.nan_to_num( k**2 * (-2*U*J1(k*np.sqrt(x**2 + y**2)))/(k*J0(k*R_dipole))*y/np.sqrt(x**2+y**2), nan=0 ) * (1.0*(np.sqrt(x**2 + y**2) < R_dipole))

N = 1000 #dipole evo was done with 2000 points
R = 0.1

# Sample radius with sqrt for uniform distribution in area
r = R * np.sqrt(np.random.uniform(0, 1, N))

# Sample angle uniformly [0, 2pi)
theta = 2 * np.pi * np.random.uniform(0, 1, N)

# Convert to Cartesian coordinates
xc = r * np.cos(theta)
yc = r * np.sin(theta)

#pv7 = PointVortex(np.array([xc,yc]), lamb_dipole(xc, yc), 0.0001)
#pv7.point_animate_circle(runs = 100, interval = 50, skip_frame=1) 
#pv7.circle_snapshots(runs=1000, skip_frame=1, num_snapshots=10, save_prefix="dipole_3000_3")
#pv7.plot_initial_condition(save_prefix="lamb_initial_cond")


