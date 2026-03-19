# -*- coding: utf-8 -*-
"""

              
"""
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix


class FVM2D(object):
    def __init__(self, Nx, Ny, hx, hy, Atype1x='2pb', Atype1y='2pb', sparse=False, xname='x', yname='y'):
        '''
        FVM2D uses finite volume method to solve partial differential equations in 2 dimensions. 
        Method of lines are used with false grid points.
        
        The grid can be seen as a matrix with false grid points outside of the domain
            
                        
                  u01 u02 u03
                  -----------                  ------------> x direction
            u10 | u11 u12 u13 | u1L            |
            u20 | u21 u22 u23 | u2L            |  
            u30 | u31 u32 u33 | u3L            |          
                  -----------                  |
                  uL1 uL2 uL3                  v  y-direction
                  
        Domain variables: u = [u11, u12, u13, u21, u22, u23, u31, u32, u33]
        False grids: uf = [u10 u20 u30 u1L u2L u3L u01 u02 u03 uL1 uL2 uL3]
                  
        Nx - number of grid points in x direction (left-right)
        Ny - number of grid points in y direction (top-down)
        hx - distance between volumes in x direction
        hy - distance between volumes in y direction
        Atype1x - Discretization method for 1st order in x direction: '2pb', '2pf', '3pc'
        Atype1y - Discretization method for 1st order in y direction: '2pb', '2pf', '3pc'
        sparse - Uses sparse matrices. Preferable for large equation systems
        xname - Creates extra methods with x replaced with xname
        yname - Creates extra methods with y replaced with yname
        
        Create an object which contains
        f = FVM2D(Nx, Ny, hx, hy, Atype1x='2pb', Atype1y='2pb', sparse=False, xname='x', yname='y')
                  
        - methods for partial derivatives
            d_dx, d_dy, d2_dx2, d2_dy2
        - methods for boundary equations (if not set, von Neumann will be used)
            set_bv_XX for XX in {x0, xL, y0, yL} 
        - methods for settings boundary values
            set_XX_values for XX in {x0, xL, y0, yL}
        - xg, yg - meshgrid matrices with x and y values
        - x, y - flattened xg and yg to be used in equations
        
        /2024 Niklas Andersson
        '''
        
        self.Nx = Nx
        self.Ny = Ny
        self.hx = hx
        self.hy = hy
        self.Atype1x = Atype1x
        self.Atype1y = Atype1y
        self.Atype2 = '3pc'
        self.sparse = sparse
        
        self.xname = xname
        self.yname = yname
        
        # Create discretization matrices for 1st order derivatives in x direction
        self.A1x = np.zeros((Nx*Ny,Nx*Ny))
        self.A1fx = np.zeros((Nx*Ny,2*(Nx+Ny)))
        if self.Atype1x=='2pb':
            self.placex(self.A1x, self.A1fx , -1/self.hx, k=-1)
            self.placex(self.A1x, self.A1fx, 1/self.hx, k=0)
        elif self.Atype1x=='2pf':
            self.placex(self.A1x, self.A1fx, -1/self.hx, k=0)
            self.placex(self.A1x, self.A1fx , 1/self.hx, k=1)
        elif self.Atype1x=='3pc':
            self.placex(self.A1x, self.A1fx, -1/(2*self.hx), k=-1)
            self.placex(self.A1x, self.A1fx , 1/(2*self.hx), k=1)
        else:
            raise Exception()
        
        # Create discretization matrices for 1st order derivatives in y direction
        self.A1y = np.zeros((Nx*Ny,Nx*Ny))
        self.A1fy = np.zeros((Nx*Ny,2*(Nx+Ny)))
        
        if self.Atype1y=='2pb':
            self.placey(self.A1y, self.A1fy , -1/self.hy, k=-1)
            self.placey(self.A1y, self.A1fy, 1/self.hy, k=0)
            
        elif self.Atype1y=='2pf':
            self.placey(self.A1y, self.A1fy, -1/self.hy, k=0)
            self.placey(self.A1y, self.A1fy , 1/self.hy, k=1)
        elif self.Atype1y=='3pc':
            self.placey(self.A1y, self.A1fy, -1/(2*self.hy), k=-1)
            self.placey(self.A1y, self.A1fy , 1/(2*self.hy), k=1)
        else:
            raise Exception()
        
        # Create discretization matrices for 2nd order derivatives in x direction
        self.A2x = np.zeros((Nx*Ny,Nx*Ny))
        self.A2fx = np.zeros((Nx*Ny,2*(Nx+Ny)))
        if self.Atype2=='3pc':
            self.placex(self.A2x, self.A2fx , 1/self.hx**2, k=-1)
            self.placex(self.A2x, self.A2fx , -2/self.hx**2, k=0)
            self.placex(self.A2x, self.A2fx, 1/self.hx**2, k=1)
        else:
            raise Exception()
        
        # Create discretization matrices for 2nd order derivatives in y direction
        self.A2y = np.zeros((Nx*Ny,Nx*Ny))
        self.A2fy = np.zeros((Nx*Ny,2*(Nx+Ny)))
        if self.Atype2=='3pc':
            self.placey(self.A2y, self.A2fy , 1/self.hy**2, k=-1)
            self.placey(self.A2y, self.A2fy , -2/self.hy**2, k=0)
            self.placey(self.A2y, self.A2fy, 1/self.hy**2, k=1)
        else:
            raise Exception()
        
        # Create discretization matrices the false grid points
        self.B1 = np.zeros((2*(self.Nx+self.Ny),self.Nx*self.Ny))
        self.B0 = np.zeros((2*(self.Nx+self.Ny)))
        
        
        # u0 is the boundary values for Dirichlet and Robin conditions
        self.u0 = np.ones_like(self.B0)
        
        # Setting default von Neumann conditions
        self.set_bv_x0(assemble=False)
        self.set_bv_xL(assemble=False)
        self.set_bv_y0(assemble=False)
        self.set_bv_yL(assemble=False)
        
        if sparse: # Convert to sparse matrices
            self.A1x = csr_matrix(self.A1x)
            self.A1fx = csr_matrix(self.A1fx)
            self.A1y = csr_matrix(self.A1y)
            self.A1fy = csr_matrix(self.A1fy)
            self.A2x = csr_matrix(self.A2x)
            self.A2fx = csr_matrix(self.A2fx)
            self.A2y = csr_matrix(self.A2y)
            self.A2fy = csr_matrix(self.A2fy)
            
            self.B1 = csr_matrix(self.B1)
            
        # Merge matrices where possible
        self.assemble()
        
        # Create support matrices/vectors of x and y values
        x = np.linspace(self.hx/2, self.Nx*self.hx-self.hx/2,self.Nx)
        y = np.linspace(self.hy/2, self.Ny*self.hy-self.hy/2,self.Ny)
        
        [xg,yg] = np.meshgrid(x,y)
        self.xg = xg
        self.yg = yg
        self.x = xg.flatten()
        self.y = yg.flatten()
        
        # Create extra methods if other independent variables than x and y
        if self.xname!='x':
            self.set_xname(self.xname)
        if self.yname!='y':
            self.set_yname(self.yname)    
        
    def set_xname(self, name='r'):
        '''
        Create extra methods for more readable code. This will create new methods
        for the new name and map them to the x and y methods.
        '''
        self.xname = name
        setattr(self, f'd_d{self.xname}', self.d_dx)
        setattr(self, f'd_d{self.xname}d{self.yname}', self.d2_dxdy)
        setattr(self, f'd2_d{self.xname}2', self.d2_dx2)
        setattr(self, f'{self.xname}profile', self.xprofile)
        setattr(self, f'set_bv_{self.xname}0', self.set_bv_x0)
        setattr(self, f'set_bv_{self.xname}L', self.set_bv_xL)
        setattr(self, f'set_{self.xname}0_values', self.set_x0_values)
        setattr(self, f'set_{self.xname}L_values', self.set_xL_values)
        setattr(self, f'{self.xname}', self.x)
        setattr(self, f'{self.xname}g', self.xg)
        
    def set_yname(self, name='z'):
        '''
        Create extra methods for more readable code. This will create new methods
        for the new name and map them to the x and y methods.
        '''
        self.yname = name
        setattr(self, f'd_d{self.yname}', self.d_dy)
        setattr(self, f'd_d{self.xname}d{self.yname}', self.d2_dxdy)
        setattr(self, f'd2_d{self.yname}2', self.d2_dy2)
        setattr(self, f'{self.yname}profile', self.yprofile)
        setattr(self, f'set_bv_{self.yname}0', self.set_bv_y0)
        setattr(self, f'set_bv_{self.yname}L', self.set_bv_yL)
        setattr(self, f'set_{self.yname}0_values', self.set_y0_values)
        setattr(self, f'set_{self.yname}L_values', self.set_yL_values)
        setattr(self, f'{self.yname}', self.y)
        setattr(self, f'{self.yname}g', self.yg)
       
    def assemble(self):
        '''
        Assemble matrices where possible for more efficient calculations
        '''
        self.A1xT = self.A1x + self.A1fx@self.B1
        self.A1yT = self.A1y + self.A1fy@self.B1
        self.A2xT = self.A2x + self.A2fx@self.B1
        self.A2yT = self.A2y + self.A2fy@self.B1
        
        self.assembleBx()
        self.assembleBy()
        
    def assembleBx(self):
        B0t = self.B0*self.u0
        
        self.B1xT = self.A1fx@B0t
        self.B2xT = self.A2fx@B0t
                
    def assembleBy(self):
        B0t = self.B0*self.u0
        
        self.B1yT = self.A1fy@B0t
        self.B2yT = self.A2fy@B0t
            
            
    
    def xprofile(self, values):
        '''
        Takes a profile in x and copies it in y direction
        
        Example: f.xprofile([1,2,3])  (Nx=3,Ny=3)
        xt = [[1,2,3],
              [1,2,3],
              [1,2,3]]
        x = [1,2,3,1,2,3,1,2,3]
            
        '''
        x0 = values
        y0 = np.linspace(0, 1, self.Ny)
        
        [xt,yt] = np.meshgrid(x0,y0)
        x = xt.flatten()
        #y = yt.flatten()
        return x
    
    def yprofile(self, values):
        '''
        Takes a profile in y and copies it in x direction
        
        Example: f.yprofile([1,2,3])  (Nx=3,Ny=3)
        xt = [[1,1,1],
              [2,2,2],
              [3,3,3]]
        x = [1,1,1,2,2,2,3,3,3]
            
        '''
        x0 = np.linspace(0, 1, self.Nx)
        y0 = values
        
        [xt,yt] = np.meshgrid(x0,y0)
        #x = xt.flatten()
        y = yt.flatten()
        return y
            
    def set_bv_x0(self, bt=1, b1=0, b0=0, assemble=True):
        '''
        Set boundary condition for x0 (left side)
        General form: bt*dudx| = b1*u| + b0
        
        Example Dirichlet: u| = uin  -->  0*dudx| = 1*u| - 1*uin
        f.set_bv_x0(bt=1, b1=1, b0=-1)  # b0 can be set to -1 and uin can be multiplied with:
        f.set_x0_values(uin) 
        
        Example von Neumann: dudx| = 0  -->  1*dudx| = 0*u| + 0
        f.set_bv_x0(bt=0, b1=0, b0=0)
        
        Example Robin: Dax*dudx| = k*(u|z - uin)  -->  Dax*dudx| = k*u|z - k*uin
        f.set_bv_x0(bt=Dax, b1=k, b0=-k)  # b0 can be set to only -k and uin can be multiplied with:
        f.set_x0_values(uin) 
        '''
        
        alpha0 = (2*bt-self.hx*b1)/(2*bt+self.hx*b1)
        beta0 = -(2*self.hx*b0)/(2*bt+self.hx*b1)
        
        
        for i in range(self.Ny):
            row = i
            col = self.Nx*i
            self.B1[row,col] = alpha0
        self.B0[:self.Ny] = beta0
        
        # if self.sparse:
        #     self.B0 = csr_matrix(self.B0)
        
        if assemble:
            self.assemble()
        
        
    def set_bv_xL(self, bt=1, b1=0, b0=0, assemble=True):
        '''
        Set boundary condition for xL (right side)
        General form: bt*dudx| = b1*u| + b0
        
        See examples in set_bv_x0
        '''
        
        alphaL = (2*bt+self.hx*b1)/(2*bt-self.hx*b1 )
        betaL = (2*self.hx*b0)/(2*bt-self.hx*b1 )
        
        for i in range(self.Ny):
            row = self.Ny + i
            col = self.Nx*(i+1)  - 1
            self.B1[row,col] = alphaL
        self.B0[self.Ny:2*self.Ny] = betaL
        
        # if self.sparse:
        #     self.B0 = csr_matrix(self.B0)
        
        if assemble:
            self.assemble()
        
    def set_bv_y0(self, bt=1, b1=0, b0=0, assemble=True):
        '''
        Set boundary condition for y0 (top side)
        General form: bt*dudy| = b1*u| + b0
        
        See examples in set_bv_x0 (Note that we now have dudy| instead of dudx|)
        '''
        alpha0 = (2*bt-self.hy*b1)/(2*bt+self.hy*b1)
        beta0 = -(2*self.hy*b0)/(2*bt+self.hy*b1)
        
        
        for i in range(self.Nx):
            row = 2*self.Ny + i
            col = i
            self.B1[row,col] = alpha0
        self.B0[2*self.Ny:2*self.Ny+self.Nx] = beta0  
        
        # if self.sparse:
        #     self.B0 = csr_matrix(self.B0)
        
        if assemble:
            self.assemble()
        
    def set_bv_yL(self, bt=1, b1=0, b0=0, assemble=True):
        '''
        Set boundary condition for yL (bottom side)
        General form: bt*dudy| = b1*u| + b0
        
        See examples in set_bv_x0 (Note that we now have dudy| instead of dudx|)
        '''
        alphaL = (2*bt+self.hy*b1)/(2*bt-self.hy*b1 )
        betaL = (2*self.hy*b0)/(2*bt-self.hy*b1 )
        
        
        for i in range(self.Nx):
            row = 2*self.Ny + self.Nx + i
            col = (self.Ny-1)*self.Nx + i
            self.B1[row,col] = alphaL
        self.B0[2*self.Ny+self.Nx:] = betaL
        
        # if self.sparse:
        #     self.B0 = csr_matrix(self.B0)
        
        if assemble:
            self.assemble()
        
    def set_x0_values(self, values):
        '''
        Set boundary values when using Dirichlet or Robin. values is a vector with
        the size of the boundary (Ny) or just a constant value for the whole side
        '''
        self.u0[:self.Ny] = values 
        
        self.assembleBx()
            
        
    def set_xL_values(self, values):
        '''
        Set boundary values when using Dirichlet or Robin. values is a vector with
        the size of the boundary (Ny) or just a constant value for the whole side
        '''
        self.u0[self.Ny:2*self.Ny] = values
        
        self.assembleBx()
        
    def set_y0_values(self, values):
        '''
        Set boundary values when using Dirichlet or Robin. values is a vector with
        the size of the boundary (Nx) or just a constant value for the whole side
        '''
        self.u0[2*self.Ny:2*self.Ny+self.Nx] = values
        
        self.assembleBy()
    
    def set_yL_values(self, values):
        '''
        Set boundary values when using Dirichlet or Robin. values is a vector with
        the size of the boundary (Nx) or just a constant value for the whole side
        '''
        
        self.u0[2*self.Ny+self.Nx:] = values
        
        self.assembleBy()
        
    def d_dx(self, u):
        '''
        Operator d_dx on domain variable u
        du_dx = f.d_dx(u) 
        Here u and du_dx are vectors of size Nx*Ny
        '''
        
        dudx = self.A1xT@u + self.B1xT
        
        return dudx
    
    def d_dy(self, u):
        '''
        Operator d_dy on domain variable u
        du_dy = f.d_dy(u) 
        Here u and du_dy are vectors of size Nx*Ny
        '''
        
        dudy = self.A1yT@u + self.B1yT
        
        return dudy
        
    def d2_dxdy(self,u):
        '''
        Operator d2_dxdy on domain variable u
        d2u_dxdy = f.d2_dxdy(u) 
        Here u and d2u_dxdy are vectors of size Nx*Ny
        '''
        
        d2u_dxdy = self.d_dy(self.d_dx(u))
            
        return d2u_dxdy
    
    def d2_dx2(self, u):
        '''
        Operator d2_dx2 on domain variable u
        d2u_dx2 = f.d2_dx2(u) 
        Here u and d2u_dx2 are vectors of size Nx*Ny
        '''
        d2udx2 = self.A2xT@u + self.B2xT
        
        return d2udx2
    
    def d2_dy2(self, u):
        '''
        Operator d2_dy2 on domain variable u
        d2u_dy2 = f.d2_dy2(u) 
        Here u and d2u_dy2 are vectors of size Nx*Ny
        '''
        
        d2udy2 = self.A2yT@u + self.B2yT
        
        return d2udy2
    
    
    def get_fp_nearest(self):
        '''
        Returns a matrix that gives the nearest domain point for every false
        point with 
        
        u_inside = Bf@u
        '''
        
        Bf = np.zeros((2*(self.Nx+self.Ny),self.Nx*self.Ny))
        
        for i in range(self.Ny):
            row = i
            col = self.Nx*i
            Bf[row,col] = 1
        for i in range(self.Ny):
            row = self.Ny + i
            col = self.Nx*(i+1)  - 1
            Bf[row,col] = 1
        for i in range(self.Nx):
            row = 2*self.Ny + i
            col = i
            Bf[row,col] = 1
        for i in range(self.Nx):
            row = 2*self.Ny + self.Nx + i
            col = (self.Ny-1)*self.Nx + i
            Bf[row,col] = 1
            
        return Bf
    
    def get_bv_values(self, u):
        '''
        Returns the values at the boundaries for a given u
        '''
        # u values at false grid points
        u_fp = self.B1@u + self.B0*self.u0
        
        # u values in the cells next to the false grid points inside the domain
        Bf = self.get_fp_nearest()
        u_inside = Bf@u
        
        # Take the average
        u_at_bnd = (u_fp + u_inside)/2
        return u_at_bnd
    
    def add_border_values(self, u):
        u_at_bnd = self.get_bv_values(u.flatten())
        
        um = self.reshape(u)
        
        x0 = u_at_bnd[:self.Ny].reshape(-1, 1) # reshape to (N,1)
        xL = u_at_bnd[self.Ny:2*self.Ny].reshape(-1, 1) # reshape to (N,1)
        y0 = u_at_bnd[2*self.Ny:2*self.Ny+self.Nx].reshape(1,-1) # reshape to (1,N)
        yL = u_at_bnd[2*self.Ny+self.Nx:].reshape(1,-1) # reshape to (1,N)
        
        # first stack left and right side
        u2 = np.hstack((x0,um,xL))
        
        # add two extra values because the matrix is wider
        y02 = np.hstack((y0[0,0] ,y0.flatten(), y0[0,-1]))
        yL2 = np.hstack((yL[0,0] ,yL.flatten(), yL[0,-1]))
         
        u_with_bv = np.vstack((y02,u2,yL2))
        
        Lx = self.Nx*self.hx
        Ly = self.Ny*self.hy
        
        x = self.xg[0,:] # take any row
        y = self.yg[:,0] # take any col
        
        # extend x and y with the boundary x and y
        xn = np.hstack((0,x,Lx))
        yn = np.hstack((0,y,Ly))
        
        return xn, yn, u_with_bv
        
        
        
    def placex(self, A, Af, value, k=0):
        '''
        Help method to place values of diagonal k in {-1,0,1} 
          for discretization matrices A and Af in x direction
        '''
        if k==0:
            for i in range(self.Nx*self.Ny):
                A[i,i] = value

        elif k==1:
            for i in range(self.Nx*self.Ny):
                if i%self.Nx==self.Nx-1:
                    continue
                else:
                    A[i,i+1] = value
                    
            for j in range(self.Ny):
                row = j*self.Nx + self.Nx - 1
                col = self.Ny + j
                Af[row,col] = value
            
        elif k==-1:
            for i in range(1,self.Nx*self.Ny):
                if i%self.Nx==0:
                    continue
                else:
                    A[i,i-1] = value
                    
            for j in range(self.Ny):
                idx = j*self.Nx
                Af[idx,j] = value
            
        else:
            raise Exception()
        

    def placey(self, A, Af, value, k=0):
        '''
        Help method to place values of diagonal k in {-1,0,1} 
          for discretization matrices A and Af in y direction
        '''
        if k==0:
            for i in range(self.Nx*self.Ny):
                A[i,i] = value
            
            
                
        elif k==-1:
            for i in range(self.Nx,self.Nx*self.Ny):
                row = i
                col = i - self.Nx
                A[row, col] = value
            
            for i in range(self.Nx):
                row = i
                col = 2*self.Ny + i
                Af[row, col] = value
            
            
                
        elif k==1:
            for i in range(self.Nx*self.Ny-self.Nx):
                row = i
                col = i + self.Nx
                A[row, col] = value
            
            
            for i in range(self.Nx):
                row = self.Nx*(self.Ny-1) + i
                col = 2*self.Ny + self.Nx + i
                Af[row, col] = value
        
        else:
            raise Exception()

    def reshape(self, u):
        um = np.reshape(u, (self.Ny, self.Nx))
        return um
            
    def FVMjpattern(self, system, N_states, y_typical=None):
        '''
        Creates a jacobian pattern to be used by matlabs odesolvers.
        Syntax: JP = FVMjpattern(lambda t,y: modelfunction(t,y, params...), N_states)
        
        N_states - the total number of states (length of yinit)
        y_typical - JPattern works best for typical values of y. Use for example
                    yinit but note that zeros are not allowed
        
        '''

        # Create a place for S
        S = np.empty((N_states,N_states))

        # Creates a random vector for the sta tes sent into the ode function.
        if y_typical is None:
            Y0 = np.random.rand(N_states)
        else:
            factor = 0.20
            yrand = 1 - factor/2 + factor*np.random.rand(N_states)
            Y0 = y_typical*yrand
            idx = np.abs(Y0)<1e-9
            N_zeros = np.sum(idx)
            Y0[idx] = np.random.rand(N_zeros)
            

        # Simulates the derivative at the base states.
        dY0 = system(0, Y0)

        # Checking all derivatives to see which state affects it.
        for i in range (N_states):

            # Reset Y
            Y = Y0.copy()

            # Changes the values of Y
            Y[i] *= 1.1

            # Runs the ode function with one state changed.
            dY = system(0,Y)

            # Checks which derivatives was affected and saves them in S as ones.
            diff2 = dY!=dY0
            S[:,i] = diff2

        return S
    
    
    def plotContour(self, ustat, add_bv=False, levels=100, aspect=1.5, ax=None, set_axis_names=True):
        '''
        Creates a contour plot using the 
        domain variables, ustat.
        
        Parameters
        ----------
        ustat : A vector of length Nx*Ny or a matrix of shape (Ny,Nx)
            The values of the domain variable (u) to be plotted
        add_bv : boolean, optional
            calculates the values at the boundaries and adds them to the plot
                The default is True
        levels : int, optional
            contour levels. The default is 100.
        aspect : TYPE, optional
            aspect ratio factor between x and y. The default is 1.5.
        ax : axes handle, optional
            axes handle to be plotted in.  If None an axis will be created. 
            The default is None.
        set_axis_names : boolean, optional
            set the names of the axis. The default is True.

        Returns
        -------
        ax : axes handle
            The axes handle of the plot

        '''
        if add_bv: # adds boundary values
            xn, yn, u_with_bv = self.add_border_values(ustat)
            xg, yg = np.meshgrid(xn, yn)
            ustat = u_with_bv
        else:
            xg = self.xg
            yg = self.yg
            ustat = np.reshape(ustat, (self.Ny, self.Nx))
            
        
        
        if ax is None:
            fig,ax = plt.subplots(1,1)
        
        h_c = ax.contourf(xg, yg, ustat, levels=levels, cmap='jet')
        ax.set_aspect(aspect)
        ax.get_figure().colorbar(h_c)
        if set_axis_names:
            plt.xlabel(f'{self.xname}')
            plt.ylabel(f'{self.yname}')
        return ax
        
    
    def set_axes_equal(self, ax):
        """Set equal aspect ratio for a 3D plot."""
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()
    
        # Calculate the midpoints and max range
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
    
        max_range = max(
            (x_limits[1] - x_limits[0]),
            (y_limits[1] - y_limits[0]),
            (z_limits[1] - z_limits[0])
        ) / 2.0
    
        ax.set_xlim([x_middle - max_range, x_middle + max_range])
        ax.set_ylim([y_middle - max_range, y_middle + max_range])
        ax.set_zlim([z_middle - max_range, z_middle + max_range])
    

    def plotAxisymmetric(self, ustat, ax=None, add_bv=True, refine=True, create_colorbar=True, set_axis_names=True):
        '''
        Creates an axisymmetric plot as an open cylinder using the 
        domain variables, ustat. The axial direction needs to be the x direction
        
        Parameters
        ----------
        ustat : A vector of length Nx*Ny or a matrix of shape (Ny,Nx)
            The values of the domain variable (u) to be plotted
        ax : axes handle, optional
            axes handle to be plotted in. If None an axis will be created. 
                The default is None.
        add_bv : boolean, optional
            calculates the values at the boundaries and adds them to the plot
                The default is True
        refine : boolean, optional
            adds interpolated values at midpoint for a finer surface
                The default is True
        
        create_colorbar : boolean, optional
            if a colorbar should be created. The default is True.
        set_axis_names : boolean, optional
            set the names of the axis. The default is True.

        Returns
        -------
        ax : axes handle
            The axes handle of the plot

        '''
        
        if add_bv: # adds boundary values
            xn, yn, u_with_bv = self.add_border_values(ustat)
            xg, yg = np.meshgrid(xn, yn)
            ustat = u_with_bv
            Nx = self.Nx + 2
            Ny = self.Ny + 2
        else:
            xg = self.xg
            yg = self.yg
            ustat = np.reshape(ustat, (self.Ny, self.Nx))
            Nx = self.Nx
            Ny = self.Ny
        
        
        if refine: # adds interpolated values at midpoints
            from scipy.interpolate import griddata
            xold = xg[0,:]
            yold = yg[:,1]
            xhalf = (xold[:-1]+xold[1:])/2
            yhalf = (yold[:-1]+yold[1:])/2
            xnew = sorted(np.hstack((xold,xhalf)))
            ynew = sorted(np.hstack((yold,yhalf)))
            xgnew, ygnew = np.meshgrid(xnew, ynew)
            
            ustat = griddata((xg.flatten(), yg.flatten()), ustat.flatten(), (xgnew, ygnew), method='linear')
            xg = xgnew
            yg = ygnew
            Ny,Nx = xg.shape
            
        Ny,Nx = ustat.shape
        
        
        umin = np.min(ustat)
        umax = np.max(ustat)
        alpha = 1 #transparency
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        angle1 = 0.1*2*np.pi # Start angle
        angle2 = 0.8*2*np.pi # Stop angle
        
        if 1: # This creates the inner sides of the cylinder
            for angle in [angle1,angle2]:
                x = xg
                y = yg*np.sin(angle)
                z = yg*np.cos(angle)
                
                utotn = (ustat-umin)/(umax-umin)
                
                facecolors = plt.cm.jet(utotn)
                ax.plot_surface(x,y,z, facecolors=facecolors, shade=False, alpha=alpha, zorder=100)
        if 0: # This creates the bottom part of the cylinder (x0)
            xt = []
            yt = []
            zt = []
            ut = []
            for angle in np.linspace(angle1,angle2,20):
                x = np.min(xg)*np.ones(Ny)
                y = yg[:,0]*np.sin(angle)
                z = yg[:,0]*np.cos(angle)
                xt.append(x)
                yt.append(y)
                zt.append(z)
                ut.append(ustat[:,0])
            x = np.vstack(xt)
            y = np.vstack(yt)
            z = np.vstack(zt)
            u = np.vstack(ut)
            utot = u
            utotn = (utot-umin)/(umax-umin)
    
            facecolors = plt.cm.jet(utotn)
            ax.plot_surface(x,y,z, facecolors=facecolors, shade=False, alpha=alpha, zorder=10)
        if 1: # This creates the top part of the cylinder (xL)
            xt = []
            yt = []
            zt = []
            ut = []
            for angle in np.linspace(angle1,angle2,20):
                x = np.max(xg)*np.ones(Ny)
                y = yg[:,-1]*np.sin(angle)
                z = yg[:,-1]*np.cos(angle)
                xt.append(x)
                yt.append(y)
                zt.append(z)
                ut.append(ustat[:,-1])
            x = np.vstack(xt)
            y = np.vstack(yt)
            z = np.vstack(zt)
            u = np.vstack(ut)
            utot = u
            utotn = (utot-umin)/(umax-umin)
    
            facecolors = plt.cm.jet(utotn)
            h_c = ax.plot_surface(x,y,z, facecolors=facecolors, shade=False, alpha=alpha, zorder=10)

        Nangles = 50
        if 1: # This creates the grid around the cylinder (outer radius, yL)
            xt = []
            yt = []
            zt = []
            ut = []
            for i,angle in enumerate(np.linspace(angle1,angle2,Nangles)):
                x = xg[0,:]
                y = np.max(yg)*np.sin(angle)*np.ones(Nx)
                z = np.max(yg[:,-1])*np.cos(angle)*np.ones(Nx)
                xt.append(x)
                yt.append(y)
                zt.append(z)
                ut.append(ustat[-1,:])
                
            x = np.vstack(xt)
            y = np.vstack(yt)
            z = np.vstack(zt)
            u = np.vstack(ut)
            utot = u
            utotn = (utot-umin)/(umax-umin)
            
            facecolors = plt.cm.jet(utotn)
            option = 2
            if option==1:
                for i in range(Nangles):
                    plt.plot(x[i,:],y[i,:],z[i,:],color=facecolors[i,0,:])
                # for j in range(Nx):
                #     plt.plot(x[:,j],y[:,j],z[:,j], color=facecolors[0,j,:])
            elif option==2:
                for i in range(Nangles):
                    for j in range(Nx-1):
                        plt.plot(x[i,j:j+2],y[i,j:j+2],z[i,j:j+2],color=facecolors[i,j,:], linewidth=5)
                        
            elif option==3:
                ax.plot_surface(x,y,z, facecolors=facecolors, shade=False, alpha=alpha, zorder=-100)
            
        try:
            ax.set_aspect('equal')
        except:
            self.set_axes_equal(ax)
        
        if set_axis_names:
            plt.xlabel(f'{self.xname}')
            plt.ylabel(f'{self.yname}')
        
        if create_colorbar:
            from matplotlib.cm import ScalarMappable
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=umin, vmax=umax)
            sm = ScalarMappable(cmap='jet', norm=norm)
            cbar = plt.colorbar(sm, ax=ax)
        return ax
    
        
        

if __name__ == "__main__":
    def model(t, c, f, u, Nx, Ny, runcase):
        
        if runcase==1:
            dcdt = -u*f.d_dx(c) + 1e-6*f.d2_dx2(c)
        elif runcase==2:
            dcdt = -u*f.d_dy(c) + 1e-6*f.d2_dx2(c)
        dcdt2 = np.reshape(dcdt, (Ny,Nx))
        
        return dcdt
    
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    plt.close('all')
    Lx = 3
    Ly = 3
    Nx = 3
    Ny = 3
    hx = Lx/Nx
    hy = Ly/Ny
    u = 0.1
    runcase = 2
    c0 = 2
    
    f = FVM2D(Nx,Ny,hx,hy, Atype1x='2pb',Atype1y='2pb',xname='r')
    if runcase==1:
        f.set_bv_x0(0, 1, -1)
        f.set_x0_values(c0)
    elif runcase==2:
        f.set_bv_y0(0, 1, -1)
        f.set_y0_values(c0)
    cinit = c0*np.ones(Nx*Ny)
    tspan = [0,10]
    sol = solve_ivp(lambda t,c: model(t,c,f,u, Nx, Ny, runcase), tspan, cinit, method='BDF')
    t = sol.t
    y = sol.y.T
    
    cstat = y[-1,:]
    f.plotContour(cstat)
    