import numpy as np
import matplotlib.pyplot as plt
import sys

class Wave_1D():
    """
    1D wave equation. Solving

    c^2 d^2/dx^2 u - d^2/dt^2 u 

    > Parameters <

    >> Integration <<
    CFL: float number for Courant-Friedrich-Lewy parameter. Interval -> (0,1)
    integrator: string for time integrator choice. Choices -> [rk2, rk3, rk4, euler]
    BCs: string for boundary condition. Choices -> [periodic, outflows]
    
    >> Grid <<
    npoints: int number of discretization points in each direction
    x1min, x1max: float physical size of the grid
    num_ghost: int number of ghost points for finite differences

    >> Physics <<
    c: wave velocity

    """

    def __init__(self,
                 CFL = 0.5,
                 npoints = 100,
                 x1min = -1,
                 x1max = +1,
                 num_ghost = 2,
                 BCs = "periodic",
                 integrator = 'euler',
                 c = 1):

        ## Checks

        # CFL
        assert CFL > 0 and CFL < 1, "CFL must be comprised in the interval (0,1)"

        # x1min, x1max 
        assert x1min < x1max, "x1min must be smaller than x1max"

        ## Checks done

        self.x1min = x1min;
        self.x1max = x1max;

        self.npoints = npoints;
        
        # Define spacing
        self.dx1 = (self.x1max-self.x1min)/self.npoints
        self.num_ghost = num_ghost

        self.i_s = self.num_ghost
        self.i_f = self.num_ghost + self.npoints

        print(f"i_s = {self.i_s}")
        print(f"i_f = {self.i_f}")
        
        # Define staggered grid shifted by dx/2, with also ghosts
        self.grid_size = self.npoints + 2*self.num_ghost
        self.grid_x1 = np.arange(self.grid_size)*self.dx1-(self.x1max-self.x1min)/2-self.num_ghost*self.dx1+self.dx1*0.5


        self.c  = c

        # Determine dt from CFL condition
        self.dt = CFL*self.dx1/self.c

        self.integrator = integrator

        self.BCs = BCs

        self.oodx = 1./self.dx1



    def integrate(self, u):
        if self.integrator == "euler":
            u_new = self.euler_integrator(u)
        elif self.integrator == "rk2":
            self.rk2_integrator(u)
        elif self.integrator == "rk3":
            u_new = self.rk3_integrator(u)
        elif self.integrator == "rk4":
            u_new = self.rk4_integrator(u)
        else:
            sys.exit(f"The {self.integrator} integrator does not exist or it has not been implemented yet.")
        return u_new
    
    def euler_integrator(self, u):
        """
        Integrate with euler timestep
        """

        rhs = self.RHS(u)

        u1 = u + self.dt * rhs

        return u1

    def rk2_integrator(self, u):
        print(f"{self.integrator} integrator to be implemented.")
        return

    def rk3_integrator(self, u):
        k1 = np.zeros_like(u)
        k2 = np.zeros_like(u)
        k3 = np.zeros_like(u)

        k1 = self.RHS(u)

        k2 = self.RHS(u + self.dt * 0.5 * k1)

        k3 = self.RHS(u + 2 * k2 * self.dt - k1 * self.dt) 

        return u + 1./6. * (k1 + 4*k2 + k3) * self.dt

    def rk4_integrator(self, u):
        k1 = np.zeros_like(u)
        k2 = np.zeros_like(u)
        k3 = np.zeros_like(u)
        k4 = np.zeros_like(u)

        k1 = self.RHS(u)
        
        k2 = self.RHS(u + self.dt * k1 * 0.5)

        k3 = self.RHS(u + self.dt * k2 * 0.5)

        k4 = self.RHS(u + self.dt * k3)

        return u + 1./6. * (k1 + 2*k2 + 2*k3 + k4) * self.dt

    def RHS(self, u):
        
        rhs = np.zeros_like(u)
        

        periodic = True
        if periodic:
            for i in range(self.num_ghost):

                print(f"i = {i} corresponds to {self.i_f-self.num_ghost+i}")

                u[i,0] = u[self.i_f-self.num_ghost+i,0]
                u[i,1] = u[self.i_f-self.num_ghost+i,1]

                print(f"i = {self.i_f+i} corresponds to {self.i_s+i}")
                u[self.i_f+i,0] = u[self.i_s+i,0]
                u[self.i_f+i,1] = u[self.i_s+i,1]



        rhs[self.i_s:self.i_f,0] = u[self.i_s:self.i_f,1]
        rhs[self.i_s:self.i_f,1] = self.c * self.second_derivative(u[:,0])
        return rhs
    
    def second_derivative(self, u_array):
        
        u_2der = (- 1 * u_array[self.num_ghost-2 : self.grid_size-self.num_ghost-2] \
                 + 16 * u_array[self.num_ghost-1 : self.grid_size-self.num_ghost-1] \
                 - 30 * u_array[self.num_ghost+0 : self.grid_size-self.num_ghost+0] \
                 + 16 * u_array[self.num_ghost+1 : self.grid_size-self.num_ghost+1] \
                 -  1 * u_array[self.num_ghost+2 : self.grid_size-self.num_ghost+2]) \
                 / 12.* self.oodx*self.oodx


        return u_2der


    def run(self,
            final_time = 1.):

        u = list()
        u = np.zeros((np.size(self.grid_x1), 2))

        alpha = 2
        u[:,0] = np.exp(-alpha*self.grid_x1**2)
        u[:,1] = -2*self.grid_x1*alpha*u[:,0]

        periodic = True
        if periodic:
            for i in range(self.num_ghost):

                print(f"i = {i} corresponds to {self.i_f-self.num_ghost+i}")

                u[i,0] = u[self.i_f-self.num_ghost+i,0]
                u[i,1] = u[self.i_f-self.num_ghost+i,1]

                print(f"i = {self.i_f+i} corresponds to {self.i_s+i}")
                u[self.i_f+i,0] = u[self.i_s+i,0]
                u[self.i_f+i,1] = u[self.i_s+i,1]
        
        test_sec_der = False
        if test_sec_der:
            der_sec_anal = -2*alpha*(u[:,0]+self.grid_x1*(-2)*self.grid_x1*alpha*u[:,0])
            der_sec = self.second_derivative(u[:,0])

            plt.plot(self.grid_x1, der_sec_anal, "-bo")
            plt.plot(self.grid_x1[self.i_s:self.i_f], der_sec, "-sr")
            plt.show()

        plt.plot(self.grid_x1, u[:,0], "-o")
        plt.show()

        t = 0.
        n_iter = 0

        while t <= final_time:
            print(t)
            t = t + self.dt
            n_iter += 1

            u = self.integrate(u)
            plt.plot(self.grid_x1[self.i_s:self.i_f], u[self.i_s:self.i_f,0], "-o")
            plt.title("Time = t")
            plt.show()
            
if __name__ == "__main__":

    myWave = Wave_1D(npoints = 21,
                     x1min = -1,\
                     x1max = +1,
                     CFL = 0.5,
                     integrator = "rk4")

    myWave.run(final_time = 10.)
