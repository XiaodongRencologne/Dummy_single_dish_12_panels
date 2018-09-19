
# coding: utf-8

# In[2]:


import numpy as np;
import math
import torch as T;
import matplotlib
import matplotlib.pyplot as plt


# In[51]:


# Build model of reflector;

def reflector(List,size,Nx,Ny,f):
    '''
    :List: List of panel position x,y;
    :size: size of panel 
    :Nx Ny: sampling points on each direction;
    :f: twice of wavelength
    '''
    Np=List.shape[1];
    x=np.linspace(-1,1,Nx)*((size-size/Nx)/2);
    y=np.linspace(-1,1,Ny)*((size-size/Ny)/2); #sampling the panel; 
    
    grid=np.moveaxis(np.meshgrid(x,y),0,-1);
    x=np.reshape(grid[...,0],(1,-1)).ravel();
    y=np.reshape(grid[...,1],(1,-1)).ravel();
    
    m_x=x.repeat(Np).reshape((-1,Np))+List[0];
    m_y=y.repeat(Np).reshape((-1,Np))+List[1];
    m_z=(m_x**2+m_y**2)/2/f;
    
    return m_x,m_y,m_z; 'shape of m is sampling points*Np'


# define the field that we want to get
def field_grid(u,v,D):
    """
    :u v: is the angle range of measurment with unit of radians
    :D:   is distance between mirror and source;
    """
    
    x = D*u;
    y = D*v;
    z = D-(x**2+y**2)/D/2;
    return x,y,z;
    
#  define aperturn field of reflector;
def gauss(m_x,m_y,x_c,y_c,amp,sigma):
    """
    2D gaussian
    """
    x=m_x-x_c;
    y=m_y-y_c;
    
    A=amp*np.exp(-(x**2+y**2)/sigma**2)
    return A;

# define error  panels

def error_z_new(P,m_x,m_y,m_z,List,N):
    P=T.tensor(P);
    P=P.view((3,-1));
    a=P[0,...];
    b=P[1,...];
    c=P[2,...];
    # smapling points of panel
    Np=b.shape[0];
    m_x=m_x.reshape(-1,Np);
    m_y=m_y.reshape(-1,Np);
    m_z=m_z.reshape(-1,Np);

    dz=a + b* T.tensor(m_x-List[0]) + c * T.tensor(m_y-List[1]);
    m_z1=T.tensor(m_z) + dz;
    
    return m_z1;
    



# define the forward calculation function
"""all of the parameters have to be convert to tensor form"""
def outerMinT(x,y):
    """Calculate the outer minus 

    Equivalent to the outer product, but with the product operator
    replaced by minus
    """
    x=T.tensor(x).view(1,-1);
    
    y=T.tensor(y).view(1,-1);

    return (x-y.transpose(0,1));

def forwardT(m_x,m_y,m_z,cut_x,cut_y,cut_z, Amp, Lambda, D):
    k=2*math.pi/Lambda;
    OM=outerMinT;
    dr=T.sqrt(OM(cut_x,m_x)**2+OM(cut_y,m_y)**2+OM(cut_z,m_z)**2)-D;
    Amp=T.tensor(Amp).view(-1,1);
    
    real=T.sum(Amp*T.cos(k*dr),dim=0)
    imag=T.sum(Amp*T.sin(k*dr),dim=0)
    
    return real,imag;


# function used to get the farword calculating function with (adjustor);
def get_function(List,size,Nx,Ny,f,
                 u,v,D,
                 Lambda,sigma,Amp_0):
    # 1. build mirror
    m_x,m_y,m_z=reflector(List,size,Nx,Ny,f);
    
    # 2. given the grids of near field
    cut_x,cut_y,cut_z=field_grid(u,v,D);
    
    # 3. aperturn field:
    Amp=gauss(m_x, m_y, 0,0,Amp_0, sigma);
    
    # 4. get function of (adjuster) based on the model we build;
    
    def calculation(parameters):
        m_z1=error_z_new(parameters,m_x,m_y,m_z,List,Nx*Ny);
        real,imag=forwardT(m_x,m_y,m_z1,cut_x,cut_y,cut_z,Amp,Lambda,D);

        return real,imag;
    
    return calculation;  

# add noise on the far-field pattern
def addnoise (real,imag,mu,sigma):
    real+=T.tensor(np.random.normal(mu,sigma,list(real.shape)));
    imag+=T.tensor(np.random.normal(mu,sigma,list(imag.shape)));

    
    
# next section is used for unit-vectors fitting method
# transfer adjustor to parameters    

def adjuster2P(adjusters):
    adjusters=T.tensor(adjusters).view(3,-1);
    p=500;
    q=500;
    a= adjusters[0,...];
    b= adjusters[1,...]/p;
    c= adjusters[2,...]/q;
    parameters=T.cat((a,b,c));
    return parameters;

# define base vectors
def basevectors(forwardfunc,ad,d_z):
    #get reference pattern
    FF=forwardfunc;   
    P=adjuster2P(ad);
    real0,imag0=FF(P);
    reference=T.cat((real0,imag0));
    
    
    # get vector pattern
    ad_0=T.tensor(ad);
    ad_0=ad_0.view(1,-1)[0];
    Np=ad_0.size()[0]

    ad_0=ad_0.repeat(Np)
    
    ad_0=ad_0.view(Np,-1);
    dz=T.eye(Np,dtype=T.double)*d_z;
    
    ad_0=ad_0+dz;
    

    base_vectors=T.zeros(Np,reference.size()[0],dtype=T.double);
    
    for n in range(Np):
        
            
        p=adjuster2P(ad_0[n])
        
        real,imag=FF(adjuster2P(ad_0[n]));
        Data=T.cat((real,imag));
        base_vectors[n]=Data-reference;
    
    return reference,base_vectors;

    
           

# In[52]:


# get the calculation function

List=np.array([[-500,500,-1500,-500,500,1500,-1500,-500,500,1500,-500,500],
               [1500,1500,500,500,500,500,-500,-500,-500,-500,-1500,-1500]]);
size=1000; # size of panel
Nx=5;    # sampling poitns on x direction;
Ny=5;    # sampling points on y direction;
f=100*10**3;
D=100*10**3;
Lambda=1#mm
sigma=1200;Amp_0=1
# define u v range
Nu=Nv=21;
u=np.linspace(-1,1,Nu)*0.002;
v=np.linspace(-1,1,Nu)*0.002;
gird=np.moveaxis(np.meshgrid(u,v),0,-1);
u=gird[...,0];
v=gird[...,1];
m_x,m_y,m_z=reflector(List,size,Nx,Ny,f)
far_field=get_function(List,size,Nx,Ny,f,u,v,D,Lambda,sigma,Amp_0);

D=90*10**3;
# near field
near_field=get_function(List,size,Nx,Ny,f,u,v,D,Lambda,sigma,Amp_0);




