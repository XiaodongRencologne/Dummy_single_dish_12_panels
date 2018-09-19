
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
cell 1: define
define distance between field and reflector;
define the smapling points.
"""
D=90*10**3; # distance between source and reflector
Nu=Nv=81;   #sampling points of far-/near-field
noise=0.5;   # rms of the noise added to measurement pattern.
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
ad1=np.random.normal(0,rms_surface,(3,12)) # ad1 is the three different adjusters

print('adjusters for test data:');
print(ad1)
p=T.tensor(FM.adjuster2P(ad1));
real_meas,imag_meas=near_field(p);

# add noise to measurement data;
if noise !=0:
    FM.addnoise(real_meas,imag_meas,0,noise);

Data_meas=T.cat((real_meas,imag_meas));


# In[ ]:


"""
Cell 3:
fitting process with pyTorch package!
"""
print('>>>> fitting process with pyTorch package')
def fitfn(ad):
    p=FM.adjuster2P(ad);
    
    real,imag=near_field(p);
    r=T.cat((real_meas-real,imag_meas-imag))

    return r;
def fitfn0(ad):
    ad=T.tensor(ad, requires_grad=True);
    r=(fitfn(ad)**2).sum();
    r.backward();
    #print(r)
    return r.data.cpu().numpy(), ad.grad.data.cpu().numpy()

# define the initial adjusters position
ad=np.zeros((3,12));
start=time.clock();
x=scipy.optimize.minimize(fitfn0,ad,method="BFGS",jac=True,tol=1e-5);
elapsed = (time.clock() - start);
print("Time used:",elapsed);

print('1.fitting results: adjusters')
print(x.x.reshape(3,-1))
print('2.residual is %e'% x.fun)

error=(np.sum(np.abs(ad1-x.x.reshape(3,-1)))/36)*1000
print('3. rms of fitting adjusters positon relative to measurement adjusters: %e um'% error);


# In[ ]:


"""
Cell 4
fitting process with numpy
"""
print('>>>>fitting process with numpy')
def fitfn(ad): 
    p=FM.adjuster2P(ad);
    real,imag=near_field(p);
    r=T.cat((real_meas-real,imag_meas-imag))
    return r;

def fitfn0(ad):
    ad=T.tensor(ad, requires_grad=True);
    r=fitfn(ad)
    return r.data.cpu().numpy()#, p.grad.data.cpu().numpy()

# define the initial adjusters position
ad=np.zeros((3,12));
start=time.clock();
x=scipy.optimize.leastsq(fitfn0,ad);
elapsed = (time.clock() - start);
print("Time used:",elapsed);

print('1.fitting results: adjusters')
print(x[0].reshape(3,-1))
error=(np.sum(np.abs(ad1-x[0].reshape(3,-1)))/36)*1000
print('2. surface error of fitting adjusters positon relative to measurement adjusters: %e um'% error);

