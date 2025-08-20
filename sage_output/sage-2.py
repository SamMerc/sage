import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from uncertainties import ufloat
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.coordinates.matrix_utilities import rotation_matrix # for stellar inclination




def sage( params, planet_pixel_size, 
                  
                  wavelength, flux_hot, flux_cold,
                  
                  spot_lat, spot_long, spot_size, ve, spotnumber, 
                  fit_ldc, 
                  plot_map_wavelength):
    
    
    '''
    This function calculates the stellar contamination using a pixellation grid approach. 
    
    input: orbital parameters [params], planet pixel size [15-50], wavelength of input model spectrum, 
    flux_hot and flux_cold is flux model of clear and active photospheres, 
    spot_lat and spot_long to define position of spots, spot_size to define its size, 
    spot_number is the number of spots, 
    fit_ldc [n, custom, exotic, adrien, intensity_profile], plot_map_wavelength defines the wavelength at which the map is calculated
    
    '''
    
    N = 1000
    midpoint = params[9]
    start_time = ((params[4] + midpoint)  - 0.125)
    end_time = ( (params[4] + midpoint) + 0.125)
    time_in_JD = np.linspace(start_time, end_time, N)
    phase = np.mod((time_in_JD - midpoint)/ params[4]+0.5, 1)-0.5
    
    if len(wavelength) == 1:
        # print('A single binned flux value provided. I hope you are using a RF function')
        
        wave_interp= np.zeros(2) + wavelength
        flux_hot_interp= np.zeros(2) + flux_hot
        flux_cold_interp= np.zeros(2) + flux_cold
        f_hot = interp1d(wave_interp, flux_hot_interp, bounds_error = False, fill_value = 0.0)
        f_cold = interp1d(wave_interp, flux_cold_interp, bounds_error = False, fill_value = 0.0)

        
    elif len(wavelength) >= 1:
        
        f_hot = interp1d(wavelength, flux_hot, bounds_error = False, fill_value = 0.0)
        f_cold = interp1d(wavelength, flux_cold, bounds_error = False, fill_value = 0.0)
        
    
    
    
    # Define input parameters:
    phaseoff    = params[0]       	
    radiusratio = params[1]				
    incl        = params[2]			
    semimajor   = params[3]		
    per         = params[4]			
    u1          = params[7]			
    u2          = params[8]	
    
    mu_profile= params[10]
    I_profile= params[11]

    inc_star= params[12]
    
    # inc_z_star= params[13]
    
    ecc = params[5]
    omega_rad = (np.pi * params[6]/180.)
        
    # Convert input variables to system variables for further calculations:
    rs  = 1./semimajor
    rp  = radiusratio*rs
    
    # Select a grid size based on system size:
    # Creates a grid wich is twice the size of the planets' size in pixels * ratio of rs/rp
    n = (2.0*planet_pixel_size*(rs/rp) + 2.0*planet_pixel_size)
    grid = np.zeros((int(n),int(n)))
    
    
    # Conversion for inclination value: degrees to radians
    inclination_rad = (incl/180.)*(np.pi)

    # Finds the planets' impact parameter (= needed to find y coordinate on grid = planet_y)
    impactparam = semimajor * np.cos(inclination_rad) * ((1-ecc**2.)/(1+ecc*np.sin(omega_rad)))

    # Checks if planet eclipses star:
    if (impactparam > (1.0+(rp/rs))):
        delta_flux = np.zeros(np.size(phase))+1
        return(delta_flux)

    # Find y-coordinate of planet on star: (ypos of center of planet on grid)
    planet_y = ((impactparam)/(1.0+(rp/rs))) * ((planet_pixel_size*(rs/rp)) + planet_pixel_size)
    
    star_pixel_rad = ((rs/rp) * planet_pixel_size)
    
    # Create a grid:
    x=np.arange(int(n))-(n/2.)
    y=np.arange(int(n))-(n/2.)
    x1, y1 = np.meshgrid(x,y, sparse = True)
    
    r = np.sqrt((x1**2.0) + (y1**2.0))
    
    # Which values of the stellar grid are within the stellar (pixel) radius (find star on grid):

    x2, y2 = np.meshgrid(x/ star_pixel_rad ,y/ star_pixel_rad, sparse = False)
    starmask_rad = ((y2 >= -1.0) & (y2 <= 1.0) & (x2**2+y2**2 <= 1.0))
    
    c = 300000 #km sec^{-1}
    
    grid_new =  np.zeros((int(n),int(n)))
    grid_new[starmask_rad] = y2[starmask_rad] * (ve/ c)
    
    '''
    #Plot for viewing the wavelength shift map
    fig = plt.figure(figsize = (9,7))
    gs = gridspec.GridSpec(1,1)

    ax0=plt.subplot(gs[0,0])
    img= ax0.imshow((grid_new).T, cmap = cm.seismic, origin = 'lower')

    cbar = plt.colorbar(img, ax= ax0)
    cbar.set_label(r'$\dfrac{\Delta \lambda}{\lambda}$')
    ax0.set_xlabel('x (pixels)')
    ax0.set_ylabel('y (pixels)')
    plt.savefig('/Users/hritam/Documents/PhD/0. PhD_Papers_posters/Figures_inpaper/Figure_2/Dopplergram.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    '''
    
    starmask = (r <= star_pixel_rad)
    total_pixels = len(r[starmask])      # Inside the stellar radius
    
    bin_flux = []
    stellar_spec = []
    contamination_factor = []
    transit_depth = []
    
    
    if fit_ldc == 'single':       
        u1= np.zeros(len(wavelength))
        u2= np.zeros(len(wavelength))
        
    elif fit_ldc == 'multi-color':
        u1= np.zeros(len(wavelength)) + params[7]
        u2= np.zeros(len(wavelength)) + params[8]
               
    elif fit_ldc == 'intensity_profile':
        I_interpolated= interp1d(mu_profile[0], I_profile, bounds_error = False, fill_value = 0.0, axis=1)
        

    for i in range(len(wavelength)):
        
        lambdaa= wavelength[i]

        mu = np.cos(np.arcsin(r[starmask]/star_pixel_rad))
        
        grid[starmask] = lambdaa
        grid = grid + (grid_new * grid)
        
        star_grid = f_hot(grid)#/ np.max(photosHO[1][wave])
        
        if fit_ldc == 'single' or fit_ldc == 'multi-color':           
            star_grid[starmask] = star_grid[starmask] *  (1-u1[i]*(1-mu)-u2[i]*(1-mu)**2.0)

            
        elif fit_ldc == 'intensity_profile':
            interpolated_intensity_prof= I_interpolated(mu)
            star_grid[starmask] = star_grid[starmask] * interpolated_intensity_prof[i] 
            
    
        
        #Rotaional broadenned stellar grid
        #plt.imshow(star_grid, cmap = cm.hot , origin = 'lower', vmin = 0.0, vmax = 1.0)
        #plt.colorbar()
        #plt.title(f'Central Wavelength = {int(lambdaa)} $\AA$')
        #plt.show()
        
        star_spec = np.sum(star_grid[starmask])/ total_pixels
        stellar_spec.append(star_spec)
        
        
        if(spotnumber > 0.0):
            
            for sn in range(0, spotnumber):
                
                #adding spot parameters
                spotlong_rad = (np.pi*(spot_long[sn])/180.0)
                spotlat_rad = (np.pi*(spot_lat[sn])/180.0)
                spotsize_rad = (np.pi*(spot_size[sn])/180.0)

                # Entering Cartesian cordinate system
                sps = star_pixel_rad * np.sin(spotsize_rad)
                spx = star_pixel_rad * np.sin(spotlong_rad) * np.sin(spotlat_rad)
                spy = star_pixel_rad * np.cos(spotlat_rad)
                spz = star_pixel_rad * np.cos(spotlong_rad) * np.sin(spotlat_rad)

                spot_inCart= np.array([[spx], [spy], [spz]])
                spx, spy, spz= stellar_rotation(active_cord= spot_inCart, phase=phaseoff)
                
                # Adding stellar inclination effects
                spot_inCart= np.array([[spx], [spy], [spz]])
                spx, spy, spz= stellar_inc(stellar_inclination= (90 - inc_star)*u.deg, active_cord=spot_inCart)
                # spot_inCart= np.array([[spx], [spy], [spz]])
                # spx, spy, spz= stellar_inc_z(stellar_inclination= (90 - inc_z_star)*u.deg, active_cord=spot_inCart)


                # Converting rotated Cartesian pixels back to GCS. 
                spotlong_rad_rot= np.arctan(spx/spz)
                if spz < 0: 
                    spotlong_rad_rot= spotlong_rad_rot + np.pi
                spotlat_rad_rot= np.arccos(spy/ star_pixel_rad)

                spot = np.zeros((int(n),int(n))) + 1.0
    
                xpos1 = (spx-1.1*sps) 
                xpos2 = (spx+1.1*sps)

                ypos1 = (spy-1.1*sps)
                ypos2 = (spy+1.1*sps)

                xelements = np.arange(xpos1, xpos2)
                yelements = np.arange(ypos1, ypos2)
                #print(xelements)

                yspot_p, xspot_p = np.meshgrid(yelements, xelements)
				
                xspot_p1 = xspot_p.reshape((len(xelements)*len(xelements)),)
                yspot_p1 = yspot_p.reshape((len(yelements)*len(yelements)),)

                zspot_p1 = np.sqrt((star_pixel_rad)**2.0 - xspot_p1**2.0 - yspot_p1**2.0)

        
                # Recalculate (x,y,z)-locations to longitude and latiude (in radians):
                longi_rad = np.arctan(xspot_p1/zspot_p1)
                lati_rad  = np.arccos(yspot_p1/star_pixel_rad)
    
                # Calculate absolute difference of longitudes (radians):

                delta_lon = np.abs(spotlong_rad_rot - longi_rad)

                # Calculate central angles (= angle between spot center and point on/in box around spot):
        
                d_sigma = np.arccos(np.cos(spotlat_rad_rot) * np.cos(lati_rad) + 
                                    np.sin(spotlat_rad_rot) * np.sin(lati_rad) * np.cos(delta_lon))

                # Create masks to find all angles which are <= spotsize_rad AND which are inside the star!

                star_mask = ((xspot_p1**2.0+yspot_p1**2.0) <= star_pixel_rad**2.0)
                inspot_mask = (d_sigma <= spotsize_rad)
        
                grid_new_spot =  np.zeros((int(n),int(n))) + 1.0
                grid_new_spot[(xspot_p1[star_mask & inspot_mask].astype(int) + 
                               int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + 
                                                       int(n)/2).astype(int)] = x2[(xspot_p1[star_mask & inspot_mask].astype(int) + 
                                                                                    int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + 
                                                                                                            int(n)/2).astype(int)] * (ve/ c)

        # Define values for spot:

        #spot[xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2, yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2] = spotflux[sn]
                mu_spot = np.cos(np.arcsin(r[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)]/star_pixel_rad))
        #spot[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = (f_cold(lambdaa)/ np.max(photosHO[1][wave]))* (1-u1*(1-mu_spot)-u2*(1-mu_spot)**2.0)
                spot[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = lambdaa
                spot = spot + (spot * grid_new_spot)
        
                spot_grid = f_cold(spot)#/ np.max(photosHO[1][wave])
                if fit_ldc == 'single' or fit_ldc == 'multi-color':            
            
                    spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] * (1-u1[i]*(1-mu_spot)-u2[i]*(1-mu_spot)**2.0)
            
                elif fit_ldc == 'intensity_profile':
                    
                    interpolated_intensity_prof_spot= I_interpolated(mu_spot)
                    
                    spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] * (interpolated_intensity_prof_spot[i]) 
                

                # spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] * (1-u1[i]*(1-mu_spot)-u2[i]*(1-mu_spot)**2.0)
        
                star_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = 0.0

                star_grid = star_grid + spot_grid
        
        #plt.imshow((grid_new_spot).T, cmap = cm.seismic, origin = 'lower')
        #cbar = plt.colorbar()
        #cbar.set_label(r'$\dfrac{\Delta \lambda}{\lambda}$')
        #plt.title('wavelength shift map on spot (Rotation)')
        #plt.show()
        
        
        total_flux = np.sum(star_grid[starmask])/ total_pixels
        bin_flux.append(total_flux)
        
        resi = (star_spec/ total_flux)
        contamination_factor.append(resi)
        
        if abs(lambdaa - plot_map_wavelength) <= 10:
            star_map_out= star_grid
      
    return bin_flux, stellar_spec, contamination_factor, star_map_out


def stellar_inc(active_cord, stellar_inclination= 0.0 * u.deg):

    '''
    This function adds the effect of stellar inclination for active regions on the star. 

    Geometry: The observer is located at Z -> + np.inf. Thus, the plane of the sky is X-Y. 
    The stellar spin axis is inclined w.r.t to the y-axis. 
    So, for star_i= 90 deg. The observer in Z axis is looking at the north pole of the star. 
    While, for star_i = 0 deg. The star is spinning face-on. 

    Input: Stellar inclination [in deg] (default= 0.0 deg), 
    Cartesian cordinate of active regions (arr[x, y, z]). Be careful with the order.  

    Output: Cartesian cordinate of active regions in the inclined stellar grid. 
    '''

    rot= rotation_matrix(stellar_inclination, 'x').T # Rotation along x-axis. 

    rotated_active_cord=np.dot(rot, active_cord)
    spx= rotated_active_cord[0][0]
    spy= rotated_active_cord[1][0]
    spz= rotated_active_cord[2][0]

    return spx, spy, spz


def stellar_rotation(active_cord, phase):

    '''
    This function rotates the stellar grid with the axis of rotation set to y-axis. 
    The observer is located at Z -> np.inf. 

    Input: Cartesian coordinate of active regions (arr[x, y, z]) on the stellar grid. 
    Rotation angle [in deg]

    '''
    rot= rotation_matrix(phase, 'y').T
    rotated_active_cord= np.dot(rot, active_cord)
    spx= rotated_active_cord[0][0]
    spy= rotated_active_cord[1][0]
    spz= rotated_active_cord[2][0]

    return spx, spy, spz

def stellar_inc_z(active_cord, stellar_inclination = 0.0 * u.deg):
    
    rot = rotation_matrix(stellar_inclination, 'z').T # Rotation along z-axis.
    
    rotated_active_cord = np.dot(rot, active_cord)
    spx = rotated_active_cord[0][0]
    spy = rotated_active_cord[1][0]
    spz = rotated_active_cord[2][0]
    
    return spx, spy, spz