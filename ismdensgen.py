##################################################################
# 
#  PURPOSE: Generates HI based on Schmidt Kennicutt Law, but with
#          a slightly different technique, and probably more corr-
#          ect.
#
##################################################################

import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
from matplotlib.backends.backend_pdf import PdfPages
from astropy.coordinates import Angle
from astropy import wcs
from scipy import stats
import scipy.interpolate as interp
import scipy.integrate as integ
import ConfigParser
import random as rn

cutoff = 3.0e17
pc = 3.086e18 # in cm                            
Config = ConfigParser.ConfigParser()
Config.read('config.ini')
path_to_file = Config.get('InputFiles','path')

def cleandata(temp_data):
    temp_data = temp_data[0] if temp_data.ndim > 2.0 else temp_data
    temp_data2 = temp_data[~np.isnan(temp_data)]
    return temp_data2[np.where(temp_data2>=cutoff)]

def GordonSurveyArea():
    #WCS parameters set with the HI map    
    w = wcs.WCS(naxis=3)
    w.wcs.crpix = [4.90000000000E+02 ,4.35000000000E+02 ,1]
    w.wcs.cdelt = np.array([-2.22222227968E-03, 2.22222227968E-03, 1])
    w.wcs.crval = [2.31837507481E+01,3.05765286655E+01,1]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN","STOKES"]

    #Get y1, y2, x1, x2 of gordon coverage
    ra_cen = Angle('01h33m50.02s')
    dec_cen = Angle('30d39m36.7s')
    offset = Angle('20m')

    ra_1 = ra_cen + offset  #1 refers to 'left' of center, and ra increases towards left (check Duric's image)
    ra_2 = ra_cen - offset
    dec_1 = dec_cen - offset
    dec_2 = dec_cen + offset

    pix_up_left = w.wcs_world2pix([[ra_1.degree, dec_2.degree,4]],1)
    pix_bottom_left = w.wcs_world2pix([[ra_1.degree, dec_1.degree,4]],1)
    pix_up_right = w.wcs_world2pix([[ra_2.degree, dec_2.degree,4]],1)
    pix_bottom_right = w.wcs_world2pix([[ra_2.degree, dec_1.degree,4]],1)
    
    x,y = pix_bottom_left[0][0], pix_bottom_left[0][1]
    width = pix_bottom_right[0][0]-x
    height = pix_up_left[0][1] - y
    
    [x,y,width,height] = np.around([x,y,width,height])
    return [int(x), int(y), int(width), int(height)]

def ISMData():
    Config = ConfigParser.ConfigParser()
    Config.read('config.ini')
    path_to_file = Config.get('InputFiles','path')
    hfile = Config.get('InputFiles','hydrogenmap')
    temp_data = fits.getdata(path_to_file+hfile)
    return temp_data

def Rimage():
    #Open image from Config file
    Rfile = Config.get('InputFiles','rmap')
    lumR, headerR = fits.getdata(path_to_file+Rfile, 0, header=True)

#Convert /beam to /pixel
    idx = np.where(np.isnan(lumR))
    lumR[idx] = 0#convert all nans to 0.
    idx_rcut = np.where((lumR <= 0) | (lumR>4.0e4))
    lumR[idx_rcut] = 0

    R_bmaj = Angle('2.5e-4d') #BMAJ, most probably degrees
    R_pix = Angle('0.25s') #arcsec/pixel (under XPIXSIZE, YPIXSIZE)
    pix_factor = (3.14*(R_bmaj.degree/(2.0*R_pix.degree))**2)/np.log(2)
    lumR_perpix = lumR/pix_factor  

    return lumR_perpix
#Units of counts/s/pixels - I think 
    
def UVimage():
    FUVfile = Config.get('InputFiles', 'fuvmap')
    lumFUV, headerFUV = fits.getdata(path_to_file+FUVfile, 0, header=True)

    #Convert /beam to /pixel
    idx = np.where(np.isnan(lumFUV))
    lumFUV[idx] = 0#convert all nans to 0.
    FUV_bmaj = Angle('1.33e-3d') #BMAJ, most probably degrees
    FUV_pix = Angle('1.5s') #arcsec/pixel (under XPIXSIZE, YPIXSIZE)
    pix_factor = (3.14*(FUV_bmaj.degree/(2.0*FUV_pix.degree))**2)/np.log(2)
    lumFUV_perpix = lumFUV/pix_factor  #This is in units of counts/s/pix
    
#Convert to ergs/s
    pc_to_cm = 3.086e18
    D = 840*1.0e3*pc_to_cm
    lambda_FUV = 1516 #Angstrom
    fluxUV = lumFUV_perpix*1.4e-15  #ergs/s/cm^-2/A^-1
    L_UV = fluxUV*(4.0*3.14*D*D)*lambda_FUV

    return L_UV

def IRimage():    
    IRfile = Config.get('InputFiles', 'irmap')
    lumIR, headerIR = fits.getdata(path_to_file+IRfile, 0, header=True)
    
    #Convert /beam to /pixel
    idx = np.where(np.isnan(lumIR))
    lumIR[idx] = 0#convert all nans to 0.
    IR_bmaj = Angle('1.6667e-3d') #BMAJ, most probably degrees
    IR_pix = Angle('1.5s') #arcsec/pixel (under XPIXSIZE, YPIXSIZE)
    pix_factor = (3.14*(IR_bmaj.degree/(2.0*IR_pix.degree))**2)/np.log(2)
    lumIR_perpix = lumIR/pix_factor
    
#Convert MJy/sr to ergs/s/pix
    sqarc_to_sr = 2.35e-11 #Convert sq. arcsec to steradian
    megajy = 1.0e6  #mega jy to jy
    remov_sr = ((headerIR['CDELT2']*3600.0)**2)*sqarc_to_sr*megajy 
    S_IR = lumIR_perpix*remov_sr  #Jy/pix
    
    pc_to_cm = 3.086e18
    D = 840*1.0e3*pc_to_cm
    nu = (3.0e8)/(24*1.0e-6)  #Hz
    L_IR = S_IR*1.0e-23*(4*3.14*D*D)*nu
    
    return L_IR

def UVimage_corrected():
    corr_coeff = 3.89 #Kennicutt et. al (2012)
    return UVimage() + corr_coeff*IRimage()

def type1a_pixel_locations(size=100):
    lumR_perpix = Rimage()
    [x,y,width,height] = GordonSurveyArea()
    lumR_perpix = lumR_perpix[y:y+height+1, x:x+width+1]
    xpix = np.arange(0,lumR_perpix.shape[1]+1,1)
    ypix = np.arange(0,lumR_perpix.shape[0]+1,1)
#Calculate CDF per pixel
    p_R = lumR_perpix/np.sum(lumR_perpix)
    cdf_R = np.cumsum(p_R).reshape(p_R.shape)
#Randomly select pixels
    snrs = np.random.random(size)
    xx,yy = np.meshgrid(xpix, ypix)
    loc_x = np.zeros_like(snrs)
    loc_y = np.zeros_like(snrs)
    for i,randcdf in enumerate(snrs):
        index = zip(*np.where(cdf_R<=randcdf))
        loc_x[i] = xx[index[-1]]
        loc_y[i] = yy[index[-1]]

    return (loc_x, loc_y)

def ccsn_pixel_locations(size=100):
    """
    Returns CC SN pixel locations based on UV+IR maps
    """

    L_UV_corr = UVimage_corrected()
    [x,y,width,height] = GordonSurveyArea()
    L_UV_corr_gord = L_UV_corr[y:y+height+1, x:x+width+1]
    xpix = np.arange(0,L_UV_corr_gord.shape[1]+1,1)
    ypix = np.arange(0,L_UV_corr_gord.shape[0]+1,1)

#Calculate CDF per pixel
    p_UV = L_UV_corr_gord/np.sum(L_UV_corr_gord)
    cdf_UV = np.cumsum(p_UV).reshape(p_UV.shape)
    
#Randomly select pixels
    snrs = np.random.random(size)
    xx,yy = np.meshgrid(xpix, ypix)
    loc_x = np.zeros_like(snrs)
    loc_y = np.zeros_like(snrs)
    for i,randcdf in enumerate(snrs):
        index = zip(*np.where(cdf_UV<=randcdf))
        loc_x[i] = xx[index[-1]]
        loc_y[i] = yy[index[-1]]
        
    return (loc_x, loc_y)

def ccsn_densities(r_harray, a):
    """ 
    Returns pixel locations where CC SN explodes based on density PDFs
    """
    
    pdf_nh, bins = np.histogram(r_harray, bins=np.logspace(np.log10(3.0e17), np.log10(r_harray.max()),5000),density=True)
    
    binvals = bins[1:]-bins[:-1]
    binshift = (np.log10(bins[1]) - np.log10(bins[0]))/2.0
    bincenters = 10.0**(np.log10(bins[1:])-binshift)

    pdf_unnormed = (bincenters**a)*pdf_nh
    norm_ccsn = 1.0/np.sum(pdf_unnormed*binvals)
    pdf_cc = norm_ccsn*pdf_unnormed
  
    return  np.random.choice(bincenters, size= 5000000, p=pdf_cc*binvals)

def type1a_densities(r_harray):
    return np.random.choice(r_harray, size=5000000)

def densityplots(snia,sncc):
    binarray = np.logspace(15,23,1000)
    plt.hist(snia,bins=binarray,density=True,histtype='step')
    plt.hist(sncc,bins=binarray,density=True,color='r',histtype='step')
#    plt.text(2.0e20,6000,r'$\mathrm{p(N_H)}$',fontsize=20)
#    plt.text(1.0e21,1000,r'$\mathrm{p(N_H)N_H^{\alpha}}$',fontsize=20)
    plt.xlabel(r'$\mathrm{N_H\ [cm^{-2}]}$',fontsize=20)
    plt.ylabel(r'N',fontsize=20)
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)
    plt.tick_params(axis='y',length=8,width=2,labelsize='18',which='major')
    plt.tick_params(axis='x',length=8,width=2,labelsize='18',which='major')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1.0e-25,1.0e-21)
    plt.xlim(1.0e16,1.0e22)
    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------#                                                                                                                    
#     SUPERNOVA EJECTA MASSES AND KINETIC ENERGIES          #                                                                                                                    
#-----------------------------------------------------------#                                                                                                                   


def randomEjectaMasses(h1_1a,h1_cc):
    #Use this for actual visibility times, taking into account surface brightness cutoff.
    index = 2.35  #Salpeter
    mlow = 2.0    #M_sun
    mhigh = 30.0  #M_sun

    mass_1a = np.ones(h1_1a.size)*1.4
    norm = (1.0-index)/(mhigh**(1.0-index) - mlow**(1.0-index))
    imf = lambda m: norm*m**(-index)

    ejmasses = np.linspace(2,30,100)
    cdf = np.zeros_like(ejmasses)
    for ind,mej in enumerate(ejmasses):
        cdf[ind] = integ.quad(imf,mlow,mej)[0]
    
    if 0:
        print integ.quad(imf, mlow, mhigh)
        print cdf[-1]
    
    f = interp.interp1d(cdf,ejmasses)
    cdfnew = np.random.random(size=h1_cc.size)
    return (f(cdfnew),mass_1a)
    
def superMasses(h1_1a,h1_cc):
    #Use this for visibility time based on transition to radiative phase don't depend on ejecta mass.
    mass_cc = np.ones(h1_cc.size)*5.0
    mass_1a = np.ones(h1_1a.size)*1.4
    return (mass_cc,mass_1a)

def superEnergies(h1_1a,h1_cc):
    rv = np.random.normal(loc = 51.0, scale = 0.3, size = (h1_1a.size+h1_cc.size)) #0.4 variance ensures Hypernova, PISNs etc occur 1/1000 times "normal" CCSN (Janka 2012)     
    return 10**(rv-51.0)

def superEnergiesIaCC(size_ia = 100, size_cc = 100):
    rv1a = np.random.normal(loc = 51.0, scale = 0.1, size = size) #0.4 variance ensures Hypernova, PISNs etc occur 1/1000 times "normal" CCSN (Janka 2012)     
    rvcc = np.random.normal(loc = 51.0, scale = 0.28, size = h1_cc.size) 

    return (10**(rv1a-51.0), 10**(rvcc-51.0))

#DELETE AFTER TESTING IS DONE 03/27/16
def superEnergiesIaCC_test(h1_1a,h1_cc):
    rv1a = np.random.normal(loc = 51.0, scale = 0.1, size = h1_1a.size) #0.4 variance ensures Hypernova, PISNs etc occur 1/1000 times "normal" CCSN (Janka 2012)     
    rvcc = np.random.normal(loc = 51.0, scale = 0.4, size = h1_cc.size) 

    return (10**(rv1a-51.0), 10**(rvcc-51.0))

#-----------------------------------------------------------#                                                                                         
#   MERGING SUBROUTINE FOR BIRTH TIMES AND H-DENSITY        #                                                                                          
#-----------------------------------------------------------#                                                                                         

def create_snarray(t_1a, t_cc,mass_1a,mass_cc,h1_1a,h1_cc,ek_1a,ek_cc):
    t = np.concatenate([t_1a, t_cc])
    ek = np.concatenate([ek_1a[0:t_1a.size], ek_cc[0:t_cc.size]])
    args = np.argsort(t)
    sorted_t = t[args]
    sorted_ek = ek[args]
    
    return (sorted_t, sorted_ek)

#-----------------------------------------------------------#                                                                                 
#   POISSON PROCESS BIRTH TIME GENERATOR SUBROUTINE         #                                                                                            
#-----------------------------------------------------------#                                                                                           

def timegen(snrate, t_size=100):
   t = np.arange(1.0e7)
   prob = np.random.rand(t.size)
   return t[np.where(prob<=snrate)]
   
def jy_to_lum(lums,dist):
    return 1.2*(1.0e24)*lums*dist*dist

def arcsec_to_parsec(diam, dist):
    #dist is in Mpc, diam is in arcsec
    fact = np.pi/(180.*3600.)
    return fact*dist*1.0e6

#a = type1a_densities()
#b = ccsn_densities(1.4)
#densityplots(a,b)

#np.savetxt('trumck_cc_dens.txt',ccsn_densities())
#np.savetxt('trumck_ia_dens.txt',type1a_densities())
