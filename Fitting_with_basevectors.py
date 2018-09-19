
# coding: utf-8

# In[ ]:


import numpy as np
import torch as T
import scipy; import scipy.optimize;
import time;

import matplotlib;
import matplotlib.pyplot as plt

import  forward_method as FM;


# In[ ]:


"""
cell 1: model of 
define distance between field and reflector;
define the smapling points.
"""
D=90*10**3; # distance between source and reflector
Nu=Nv=41;   #sampling points of far-/near-field
noise=0.5;  # rms of the noise added to measurement pattern.
rms_surface=0.08#mm # rms of the surface error.

Nx=5;    # sampling poitns on x direction; for panels
Ny=5;    # sampling points on y direction;
##############################################################################
List=np.array([[-500,500,-1500,-500,500,1500,-1500,-500,500,1500,-500,500],
               [1500,1500,500,500,500,500,-500,-500,-500,-500,-1500,-1500]]);
size=1000; # size of panel
Nx=5;    # sampling poitns on x direction;
Ny=5;    # sampling points on y direction;
f=100*10**3;

Lambda=1#mm
sigma=1200;Amp_0=1
# define u v range

u=np.linspace(-1,1,Nu)*0.002;
v=np.linspace(-1,1,Nu)*0.002;
gird=np.moveaxis(np.meshgrid(u,v),0,-1);
u=gird[...,0];
v=gird[...,1];
m_x,m_y,m_z=FM.reflector(List,size,Nx,Ny,f)
far_field=FM.get_function(List,size,Nx,Ny,f,u,v,f,Lambda,sigma,Amp_0);

# near field
near_field=FM.get_function(List,size,Nx,Ny,f,u,v,D,Lambda,sigma,Amp_0);


# In[ ]:


"""
Cell 2: get the measurement data!
"""
# get measurement data;

ad1=np.random.normal(0,rms_surface,(3,12));

p=T.tensor(FM.adjuster2P(ad1));

real_meas,imag_meas=near_field(p);

if noise != 0:
    FM.addnoise(real_meas,imag_meas,0,noise);# add noise with measurement data;


Data_meas=T.cat((real_meas,imag_meas));


# In[ ]:


"""
Cell 3:
fitting process with unit_vectors
"""

start=time.clock();
ad=np.zeros((3,12));
dz=10/1000;
for n in range(5):
    reference,base_vectors=FM.basevectors(near_field,ad,dz);
    test=Data_meas-reference;
    def fitfn(p):
        
        #p=T.tensor(p, requires_grad=True);        
        p=T.tensor(p).view(-1,1);
        r=(test-T.sum(p*base_vectors,0));
        return r.data.cpu().numpy()#,p.grad.data.cpu().numpy()
    para=np.zeros((3,12));
    x=scipy.optimize.leastsq(fitfn,para);
    #print(x[0])
    ad=ad+x[0].reshape((3,12))*dz;
    dz=1/1000;
    
elapsed = (time.clock() - start);
print("Time used:",elapsed);   
error=np.sum(np.abs(ad-ad1))/36*1000;
print('1. fitting error: %e um'% error);
print('2. fitting results:');
print(ad);
print('3. real adjusters of test:')
print(ad1);

    

