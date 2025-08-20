import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import warnings
import emcee
from scipy.stats import norm, uniform, truncnorm
import multiprocessing as mp
mp.set_start_method(method="fork", force=True)
from multiprocessing import Pool
import corner
import argparse
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
from PyAstronomy.pyasl import binningx0dt

# Loading sage. 
import sys
sys.path.append('/home/users/c/chakrabh/sage/sage_model')
import sage
from sage import sage_class

# My Multi-processing packages. 
import argparse
parser = argparse.ArgumentParser(description='[description of what your code does, although this isn’t really important]')
 
parser.add_argument('-c', action="store", dest="numcores", type=int,
                    default=20, help='Number of cores to use for running') 
args = parser.parse_args()
numcores = args.numcores


nbins = 500
sector_no = 38
spotnumber = 3

backend_output = '/home/users/c/chakrabh/sage/hip67522/mcmc_output_final'
backend_name = f'hip67522_n{nbins}_lc_{sector_no}_{spotnumber}spots'

############## NO EDITING BEYOND THIS POINT ##############

pdc_sap= np.loadtxt(f'/home/users/c/chakrabh/sage/hip67522/lightcurve/sector-{sector_no}.txt').T

ti= pdc_sap[0][np.isfinite(pdc_sap[1])] 


# Removing transits of planet b. 
T0b = 2458604.02432023 
periodb = 6.95946702 
T14b = 0.20271346 

n_low  = np.round((min(ti) - T0b)/periodb, 0)
n_high = np.round((max(ti) - T0b)/periodb, 0)

n = np.arange(n_low, n_high+1, 1)
predicted_T0s = T0b + n*periodb


inds_intransit_alles = []
for predicted_T0 in predicted_T0s:
    
    inds_intransit = np.where((ti<predicted_T0-(T14b/2)) | (ti>predicted_T0+(T14b/2)))
    
    inds_intransit_alles.append(inds_intransit)

common_values = inds_intransit_alles[0]
for arr in inds_intransit_alles[1:]:
    common_values = np.intersect1d(common_values, arr)
    
    
ti= pdc_sap[0][np.isfinite(pdc_sap[1])] - min(pdc_sap[0][np.isfinite(pdc_sap[1])])
fl= pdc_sap[1][np.isfinite(pdc_sap[1])]
fl_err = pdc_sap[2][np.isfinite(pdc_sap[1])]

ti, fl, fl_err = ti[common_values], fl[common_values], fl_err[common_values]


# inds = np.where(ti>13)
# ti, fl, fl_err = ti[inds], fl[inds], fl_err[inds]

r1, dt1 = binningx0dt(ti, fl, fl_err, nbins=nbins)

bti = r1[::,0]
bfl = r1[::,1]
bfl_err = r1[::,2]

plt.errorbar(ti, fl, fl_err)
plt.errorbar(bti, bfl, bfl_err, fmt='ko')
plt.savefig(f'{backend_name}_lightcurve.png')
plt.close()

def differential_rotation_model(phi, p_rot, shear):
    '''This is the updated differential rotation model.'''
    A = 360/p_rot
    B = -2.504 
    C = -2.23  
    return A+ shear*(B*(np.sin(np.deg2rad(phi)))**2+C*(np.sin(np.deg2rad(phi)))**4)

# def differential_rotation_model(phi, p_rot, shear):
#     A = 360/p_rot
#     B = -3.1238087557001792
#     C = 0.6539271078695194
#     return A+ shear*(B*(np.sin(np.deg2rad(phi)))**2+C*(np.sin(np.deg2rad(phi)))**4)


############### BLOCK-2 #################
# defining the paramater space
if sector_no == 11:
    print('Initial spot parameters for sector 11')
    # spot_lat = [24, 31, 55] # in degress
    # spot_longitude=   [-133, 17, 96] # in degress
    # spot_size = [17, 15, 14] # in degress 
    
    spot_lat = [15, -28, 51] # in degress
    spot_longitude=   [-100, 43, 168] # in degress
    spot_size = [20, 32, 20] # in degress 
    
    
elif sector_no == 38:
    print('Initial spot parameters for sector 38')
    # spot_lat = [45, -21, 24] # in degress
    spot_lat = [10, -10, 10] # in degress
    spot_longitude=   [-155, 20, 110] # in degress
    # spot_size = [38, 17, 36] # in degress   
    spot_size = [8, 8, 8] # in degress   

elif sector_no == 64:
    print('Initial spot parameters for sector 64')
    spot_lat = [54, -22, 41]
    spot_longitude=   [-69, 114, 145]
    spot_size = [27, 27, 36]


# emcee
spot_params=[]
spot_priors= []
spot_labels= []
for num in range(spotnumber):
    
    # start value
    lat= spot_lat[num]
    long= spot_longitude[num]
    siz= spot_size[num]
    spot_params.extend([lat, long, siz])

    # priors
    if num % 2 == 0:
        lat_= uniform(0, 80)
    else:
        lat_= uniform(-80, 80)
    
    # lat_= uniform(0, 70) # Putting smart constraints in the latitude and on the spot_lat
    long_= uniform(-180, 360)
    # siz_= uniform(1, 41)
    siz_= norm(8, 1)
    spot_priors.extend([lat_, long_, siz_])

    
    # labels
    lat_label= f'sLat_{num}'
    long_label= f'sLong_{num}'
    siz_label= f'sSize_{num}'
    spot_labels.extend([lat_label, long_label, siz_label])

params= spot_params
params.append(0.0187)    # Added a parameter for offset
params.append(-10)        # Added a parameter for jitter
params.append(1.38)    # Added a parameter for rotation period
params.append(5.)      # Added a parameter for shear factor

priors= spot_priors
priors.append(uniform(-1.5, 2.5))     # priors for offset
priors.append(uniform(-15, 15))     # priors for jitter
priors.append(norm(1.38, .05))     # priors for rotation period
priors.append(uniform(0, 10.5))    # priors for shear factor
# priors.append(truncnorm( a= (0-32)/15, b=(90-32)/15, loc=32, scale=15 ))  # priors for inclination

spot_labels.append("offset")
spot_labels.append("jitt")
spot_labels.append("Prot")
spot_labels.append("shear")
# spot_labels.append("i$_{\star}$")

################# BLOCK-3 #################

# defining MCMC funcs (log prior, log likelihood, log probability)
def lnprior(params,priors=priors):
    lp = 0.0
    for par, pr in zip(params, priors):
        lp += pr.logpdf(par)
    return lp

def lnlike(params, time, flux, flux_err):
    
    # disintegrating the params to make it usable in sage. 
    spot_long= []
    spot_lat=[]
    spot_size= []
    for num in np.arange(0, spotnumber*3, 3):
        lat= params[num]
        long= params[num+1]
        siz= params[num+2]
        
        spot_lat.append(lat)
        spot_long.append(long)
        spot_size.append(siz)
    
    # inclination= params[-1] # inclination of star
    inclination=90
    offset= params[-4]      # offset term
    jitt= params[-3]        # jitter term
    
    # defining wavelength params
    wavelength= [5000]  # cal wavelength [pretty much useless]
    flux_hot=[1]        # immaculate photosphere
    flux_cold= [0.87] #[0.2691] # contrast

    u1= 0.32438142
    u2= 0.26073528
    
    stellar_params=[0.1,                                       # Radius-ratio   
                    20.0,                                      # scaled semi-major axis 
                    u1,                                        # U1
                    u2,                                        # U2
                    0.0,                                       # cosine of angular distance
                    0.0,                                       # Intensity profile 
                    inclination]                               
    
    planet_pixel_size= 10
    
    ve=0.0
    
    prot= params[-2]
    
    shear= params[-1]

    model_lightcurve = np.empty(len(time))
    
    for i, ti in enumerate(time):
        
        shift_perday = differential_rotation_model(spot_lat, prot, shear)
        
        spot_long1 = spot_long + (shift_perday*ti)
        
        star = sage_class(stellar_params, planet_pixel_size, wavelength, flux_hot, flux_cold, 
                 spot_lat, spot_long1, spot_size, ve, spotnumber, 'multi-color', 5000, phases_rot=[np.rad2deg(0.0)])
        
        flux_norm, contamination_factors, star_map= star.rotate_star()
        
        model_lightcurve[i] = flux_norm 
          
    mdl = model_lightcurve + offset # delete this when you are done with the code.
    
    mdl= np.asarray(mdl)
    
    sigma2= flux_err**2 + mdl**2 * np.exp(2*jitt) # the corrected error in flux.
    
    return -0.5 * np.sum((flux - mdl)**2/sigma2 + np.log(sigma2) ) 

def lnprob(params, day, flux, flux_err):
    
    lp = lnprior(params)
    
    return lp + lnlike(params, day, flux, flux_err) if np.isfinite(lp) else -np.inf

# defining the number of priors.
ndim, nwalkers = len(priors), len(priors)*5
pos = np.array([params + 1e-3*np.random.randn(ndim) for i in range(nwalkers)])
for position in pos:
    assert np.isfinite(lnprior(position,priors)), f"lnprior of parameters: {position} is not finite"
    
fig,ax = plt.subplots(1,ndim, figsize=(15,5))
ax = ax.reshape(-1)

for i,pr in enumerate(priors):
    ax[i].hist(pr.rvs(100000), bins=100)
    ax[i].axvline(position[i],color="red")
    ax[i].set_xlabel(spot_labels[i])
plt.savefig('prior.png')
plt.tight_layout()
plt.show()

if __name__ == '__main__': 
    reader = emcee.backends.HDFBackend(f'{backend_output}/{backend_name}.h5')
    reader.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(bti, bfl, bfl_err), pool=Pool(numcores) , backend= reader)
    sampler.run_mcmc(pos, 5000, progress=True);