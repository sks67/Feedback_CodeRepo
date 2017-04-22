#-----------------------------------------------------------#
# OFFICIAL DRIVER SCRIPT (08/13/16)
#-----------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as py
import scipy.interpolate as interp
import scipy.integrate as integ
import trumck_nvst_3 as trumck
import ismdensgen as dgen
reload(trumck)
reload(dgen)
import time as tm
import datetime as dt
import os
import sys

frac = 0.1

if 0:
    thick_array = np.array([100.])*1.0
    ratelmc = np.array([ 5.0e-3])
    ratefactor = np.array([1.])
    epse_array = np.array([0.005])

if 1:
    thick_array = np.linspace(0,500,10)
    ratelmc = np.linspace(1.0,7.0, 10)*1.0e-3
    #ratelmc = np.logspace(-4,-1,10)
    ratefactor = np.ones(10)*15.0# np.array([15.,15.,15.,15.,15.,15.,10.,3.,2.,1.])
    epse_array = np.logspace(-5,-1, 10)
    
likhood_lum = np.zeros(thick_array.size*ratelmc.size*epse_array.size)
likhood_diam = np.zeros_like(likhood_lum)

xx=0
start = tm.time()

locx_IA = np.loadtxt('randompixels_x_ia.txt').astype(int)
locy_IA = np.loadtxt('randompixels_y_ia.txt').astype(int)
locx_CC = np.loadtxt('randompixels_x.txt').astype(int)
locy_CC = np.loadtxt('randompixels_y.txt').astype(int)

hmap = dgen.ISMData()[0]
[x,y,width,height] = dgen.GordonSurveyArea()
hmap_Gord = hmap[y:y+height+1, x:x+width+1]


h1_ia_pdf = hmap_Gord[locy_IA, locx_IA]
h1_cc_pdf = hmap_Gord[locy_CC, locx_CC]


#Create a larger set
h1_1a = np.random.choice(h1_ia_pdf, size=5000000)
h1_cc = np.random.choice(h1_cc_pdf, size=5000000)

#Remove negative values
h1_1a = h1_1a[np.where(h1_1a>0)]
h1_cc = h1_cc[np.where(h1_cc>0)]


#mass_cc,mass_1a = dgen.randomEjectaMasses(h1_1a,h1_cc)
mass_cc,mass_1a = dgen.superMasses(h1_1a,h1_cc)
ek_1a, ek_cc  = dgen.superEnergiesIaCC(h1_1a,h1_cc)

for j in range(thick_array.size):  #Scale Height
    print j
    for idx, rate_elem in enumerate(ratelmc): #SN Rate
        rate_elem = rate_elem*ratefactor[idx]
        rate_1a_lmc = (frac/(1+frac))*rate_elem 
        rate_cc_lmc = (1/(1+frac))*rate_elem
        time_1a = dgen.timegen(rate_1a_lmc)
        time_cc = dgen.timegen(rate_cc_lmc)
        for epse in epse_array: #epse
#            print 'rate = ', ratelmc, ' epse = ', epse
            if thick_array[j]==0.0 or rate_elem == 0.0 or epse==0.0:
                likhood_lum[xx],likhood_diam[xx] = 0.0,0.0
                xx=xx+1
                continue
            time,h1,ejmas,energ,nprof = dgen.create_snarray(time_1a,time_cc,mass_1a,mass_cc,h1_1a,h1_cc,ek_1a, ek_cc)
            likhood_lum[xx],likhood_diam[xx] = trumck.likelihoods(thick_array[j],epse,h1,time,ejmas,energ,nprof, rateFactor=ratefactor[idx])
                
            xx=xx+1

print 'RUNTIME = ',str(dt.timedelta(seconds=tm.time()-start))
print likhood_lum
print likhood_diam

if 1:        
    userdoc = os.path.join(os.getcwd(),'DataAnalysis')                                                                                                      
    np.savetxt(os.path.join(userdoc,'ParameterSpacePlot_Fraction01.txt'),likhood_lum)
#np.savetxt(os.path.join(userdoc,'epsb_likhood_diam_MultWavmaps.txt'),likhood_diam)
#np.savetxt(os.path.join(userdoc,'epsb_likhood_dens_lowercut.txt'),likhood_dens)

#np.savetxt(os.path.join(userdoc,'checksnaps_easier_thickness.txt'),thick_array)
#np.savetxt(os.path.join(userdoc,'checksnaps_easier_fraction.txt'),frac_array)
#np.savetxt(os.path.join(userdoc,'checksnaps_easier_snrate.txt'),ratelmc)
