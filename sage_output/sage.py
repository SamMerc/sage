import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.coordinates.matrix_utilities import rotation_matrix # for stellar inclination

class sage_class:
    
    def __init__(self, params, planet_pixel_size, wavelength,
                 flux_hot, flux_cold, 
                 spot_lat, spot_long, spot_size, ve, spotnumber, 
                 fit_ldc,
                 plot_map_wavelength, phases_rot= 0):
        # Assing input variables to as attributes of the class.
        self.params= params
        self.planet_pixel_size= planet_pixel_size
        self.wavelength= wavelength
        self.flux_hot= flux_hot
        self.flux_cold= flux_cold
        self.spot_lat= spot_lat
        self.spot_long= spot_long
        self.spot_size= spot_size
        self.ve= ve
        self.spotnumber= spotnumber
        self.fit_ldc= fit_ldc
        self.plot_map_wavelength= plot_map_wavelength
        self.phases_rot= phases_rot
        
    def rotate_star(self):

        if len(self.phases_rot) == 1:
            self.phases_rot= self.phases_rot[0] # The 0 is there to take the first entry. 
            # print('No rotation')
            lc, epsilon_wl, star_maps= self.StarSpotSpec()
            
        elif len(self.phases_rot) != 1:
            # print('Rotating the star')
            lc= []
            epsilon_wl= []
            star_maps= []
            for i, n in enumerate(self.phases_rot):
                self.phases_rot= n
                flux_norm, contamination_factor, star_map= self.StarSpotSpec()
                lc.append(flux_norm)    
                epsilon_wl.append(contamination_factor)
                star_maps.append(star_map)
            lc= lc/np.median(lc) # added to remove the peak flux 1.0 case.   #TODO discuss with ML about what's best way of normalising. clear stellar spec or median of lc.                         

        return lc, epsilon_wl, star_maps
                
        
        
    def StarSpotSpec(self):
        
        '''
        This function calculates the stellar contamination using a pixellation grid approach as presented in chakraborty et al. (in prep). 
        
        input: orbital parameters [params], planet pixel size [15-50], wavelength of input model spectrum, 
        flux_hot and flux_cold is flux model of clear and active photospheres, 
        spot_lat and spot_long to define position of spots, spot_size to define its size, 
        spot_number is the number of spots, 
        fit_ldc [n, custom, exotic, adrien, intensity_profile], plot_map_wavelength defines the wavelength at which the map is calculated
        
        '''

        if len(self.wavelength) == 1:
            wave_interp= np.zeros(2) + self.wavelength
            flux_hot_interp= np.zeros(2) + self.flux_hot
            flux_cold_interp= np.zeros(2) + self.flux_cold
            f_hot = interp1d(wave_interp, flux_hot_interp, bounds_error = False, fill_value = 0.0)
            f_cold = interp1d(wave_interp, flux_cold_interp, bounds_error = False, fill_value = 0.0)
            
        else:
            
            f_hot = interp1d(self.wavelength, self.flux_hot, bounds_error = False, fill_value = 0.0)
            f_cold = interp1d(self.wavelength, self.flux_cold, bounds_error = False, fill_value = 0.0)
        
        # The input parameters for grid:
        
        # for rotation phases
        phaseoff= self.phases_rot
        # for grid-size
        radiusratio = self.params[0]						
        semimajor   = self.params[1]			
        
        mu_profile= self.params[4]
        I_profile= self.params[5]
        # for stellar inclination
        inc_star= self.params[6]

        # Converting latitude to co-latitude
        spot_lat= 90 - np.asarray(self.spot_lat)
            
        # Convert input variables to system variables for further calculations:
        rs  = 1./semimajor
        rp  = radiusratio*rs
      
        ### TIME CONSUMING ###
        # Select a grid size based on system size:
        # Creates a grid wich is twice the size of the planets' size in pixels * ratio of rs/rp
        n = int(2.0*self.planet_pixel_size*(rs/rp) + 2.0*self.planet_pixel_size)
        n2 = n//2
        
        star_pixel_rad = ((rs/rp) * self.planet_pixel_size)
        star_pixel_rad2 = float(star_pixel_rad)**2

        # Create a grid:
        x = (np.arange(n, dtype=np.int64) - n2)
        y = (np.arange(n, dtype=np.int64) - n2)
        
        r = np.sqrt(x[None, :]**2 + y[:, None]**2).astype(np.float64)
        r2 = r*r

        # Which values of the stellar grid are within the stellar (pixel) radius (find star on grid):
        x2 = (x / star_pixel_rad)[None, :].repeat(n, axis=0).astype(np.float32)
        y2 = (y / star_pixel_rad)[:, None].repeat(n, axis=1).astype(np.float32)
        starmask_rad = ((y2 >= -1.0) & (y2 <= 1.0) & ( ( (x2*x2) + (y2*y2) ) <= 1.0) )
        
        c = 299792 #km sec^{-1}

        grid_new = np.zeros((n, n), dtype=np.float32)
        grid_new[starmask_rad] = y2[starmask_rad] * (self.ve/ c)
        
        starmask = (r <= star_pixel_rad)
        total_pixels = int(np.count_nonzero(starmask))      # Inside the stellar radius

        # Precompute μ on the full grid ONCE; reuse later (also float32 to save RAM)
        # μ = sqrt(1 - (r/R)^2), robust to rounding just inside the limb
        with np.errstate(invalid='ignore'):
            mu_full = np.zeros((n, n), dtype=np.float32)
            mu_full[starmask] = np.sqrt(
                np.maximum(0.0, 1.0 - ((r2[starmask] / star_pixel_rad2)))
            ).astype(np.float32)
        ######################
        bin_flux = []
        stellar_spec = []
        contamination_factor = []
        
        if self.fit_ldc == 'single':       
            u1= np.zeros(len(self.wavelength), dtype=np.float64)
            u2= np.zeros(len(self.wavelength), dtype=np.float64)        
        elif self.fit_ldc == 'multi-color':
            u1= np.zeros(len(self.wavelength), dtype=np.float64) + self.params[2]
            u2= np.zeros(len(self.wavelength), dtype=np.float64) + self.params[3]               
        elif self.fit_ldc == 'intensity_profile':
            I_interpolated= interp1d(mu_profile[0], I_profile, bounds_error = False, fill_value = 0.0, axis=1)

        # Flatten star indices once (avoids repeated mask logic)
        iy_star, ix_star = np.nonzero(starmask)
        mu = mu_full[iy_star, ix_star]        # (M,)
        vel_star = grid_new[iy_star, ix_star]      # (M,) already v/c scaling

        for i, wave in enumerate(self.wavelength):
            ### TIME CONSUMING ###
            lambdaa= float(wave)

            # Scalar spectrum at this λ for the hot photosphere
            # (grid outside star is zero due to fill_value=0)
            F_hot = float(f_hot(lambdaa))

            # Build star_grid only on the star pixels, then scatter back
            #   Base: F_hot * (1 + vel_star)  (keeps your original "flux * (1 + v/c)" behavior)
            star_vals = F_hot * (1.0 + vel_star)   # (M,)

            if self.fit_ldc in ('single', 'multi-color'):
                ld = (1.0 - u1[i]*(1.0 - mu) - u2[i]*((1.0 - mu)**2))
                star_vals *= ld
            else:
                interpolated_intensity_prof = I_interpolated(mu)   # shape (n_lambda, M) in your code
                star_vals *= interpolated_intensity_prof[i]             # (M,)

            # Assemble full 2D star grid efficiently
            star_grid = np.zeros((n, n), dtype=np.float32)
            star_grid[iy_star, ix_star] = star_vals.astype(np.float32)

            star_spec = star_grid[iy_star, ix_star].sum() / float(total_pixels)
            stellar_spec.append(star_spec)

            ######################

            if(self.spotnumber > 0.0):
                
                for sn in range(0, self.spotnumber):

                    #adding spot parameters
                    spotlong_rad = (np.pi*(self.spot_long[sn])/180.0)
                    spotlat_rad = (np.pi*(spot_lat[sn])/180.0)
                    spotsize_rad = (np.pi*(self.spot_size[sn])/180.0)

                    # Entering Cartesian cordinate system
                    sps = star_pixel_rad * np.sin(spotsize_rad)
                    spx = star_pixel_rad * np.sin(spotlong_rad) * np.sin(spotlat_rad)
                    spy = star_pixel_rad * np.cos(spotlat_rad)
                    spz = star_pixel_rad * np.cos(spotlong_rad) * np.sin(spotlat_rad)

                    spot_inCart= np.array([[spx], [spy], [spz]])
                    spx, spy, spz= stellar_rotation(active_cord= spot_inCart, phase=phaseoff) # for stellar rotation
                    
                    # Adding stellar inclination effects
                    spot_inCart= np.array([[spx], [spy], [spz]])
                    spx, spy, spz= stellar_inc(stellar_inclination= (90 - inc_star)*u.deg, active_cord=spot_inCart) # for stellar inclination

                    # Converting rotated Cartesian pixels back to GCS. 
                    spotlong_rad_rot= np.arctan2(spx, spz)
                    if spz < 0: 
                        spotlong_rad_rot= spotlong_rad_rot + np.pi
                    spotlat_rad_rot= np.arccos(spy/ star_pixel_rad)

                    xpos1 = (spx-1.1*sps) 
                    xpos2 = (spx+1.1*sps)

                    ypos1 = (spy-1.1*sps)
                    ypos2 = (spy+1.1*sps)

                    if not (np.isfinite(xpos1) and np.isfinite(xpos2)) or (xpos2 <= xpos1):
                        continue 

                    xelements = np.arange(xpos1, xpos2)
                    yelements = np.arange(ypos1, ypos2)
                    #print(xelements)

                    ### TIME CONSUMING ###
                    # Build the bounding box grid (still vectorized)
                    yspot_p, xspot_p = np.meshgrid(yelements, xelements, indexing='xy')
                    xspot_p1 = xspot_p.ravel()
                    yspot_p1 = yspot_p.ravel()

                    # Limit computations to pixels inside the stellar disk
                    in_star = (xspot_p1*xspot_p1 + yspot_p1*yspot_p1) <= star_pixel_rad2
                    if not np.any(in_star):
                        continue
                    
                    #Locate coordinates in the stellar grid
                    xs = xspot_p1[in_star]
                    ys = yspot_p1[in_star]

                    # z only where needed
                    zs = np.sqrt(np.maximum(0.0, star_pixel_rad2 - xs*xs - ys*ys))

                    # Long/lat of those pixels
                    longi_rad = np.arctan2(xs, zs)                     # safer than arctan(x/z)
                    lati_rad  = np.arccos(ys / star_pixel_rad)
        
                    # Calculate absolute difference of longitudes (radians):
                    delta_lon = np.abs(spotlong_rad_rot - longi_rad)

                    # Calculate central angles (= angle between spot center and point on/in box around spot):
            
                    d_sigma = np.arccos(
                            np.cos(spotlat_rad_rot) * np.cos(lati_rad) + 
                            np.sin(spotlat_rad_rot) * np.sin(lati_rad) * np.cos(delta_lon)
                    )

                    # Pixels that are both on the star and inside the spot
                    in_spot = (d_sigma <= spotsize_rad)
                    if not np.any(in_spot):
                        continue

                    #Locate coordinates in the spot
                    xs = xs[in_spot]
                    ys = ys[in_spot]

                    # Convert to integer image indices once
                    ix = (xs + n2).astype(np.intp)
                    iy = (ys + n2).astype(np.intp)

                    # Velocity field at those spot pixels (you used x2 for spots)
                    v_over_c = x2[iy, ix].astype(np.float64) * (self.ve / c)

                    # Doppler-shifted wavelengths for those pixels
                    lam_shift = lambdaa * (1.0 + v_over_c)

                    # Get cold spectrum at those shifted wavelengths (vectorized)
                    F_cold_vec = f_cold(lam_shift).astype(np.float64)    # shape (K,)

                    # Limb darkening factors at those pixels (use precomputed μ)
                    mu_spot = mu_full[iy, ix].astype(np.float64)

                    if self.fit_ldc in ('single', 'multi-color'):
                        ld_spot = (1.0 - u1[i]*(1.0 - mu_spot) - u2[i]*((1.0 - mu_spot)**2))
                        F_cold_vec *= ld_spot
                    else:
                        # intensity profile: interpolate only at needed μ values
                        # I_interpolated returns shape (n_lambda, M) if fed vector; pick i-th λ row
                        # To keep memory light, call it once and index
                        I_spot = I_interpolated(mu_spot)[i]
                        F_cold_vec *= I_spot

                    # Directly REPLACE hot photosphere by cold spot values at those pixels
                    star_grid[iy, ix] = F_cold_vec.astype(np.float32)            

            total_flux = star_grid[iy_star, ix_star].sum() / float(total_pixels)
            bin_flux.append(total_flux)
            
            resi = (star_spec/ total_flux) # a proof for this formula is available in the paper.
            contamination_factor.append(resi)
            
            if abs(lambdaa - self.plot_map_wavelength) <= 10:
                star_map_out= star_grid     

        # calculating drop in stellar flux due to active regions.
        spotted_flux= np.sum(bin_flux)
        unspotted_flux= np.sum(stellar_spec)
        flux_norm= spotted_flux/ unspotted_flux # Normalising by unspotted stellar flux sets the peak flux to 1.0. This might not be the case on the basis of your normalisation. 
        # flux_norm= spotted_flux#/ np.median(spotted_flux)
               
        return flux_norm, contamination_factor, star_map_out

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
    """This function rotates the stellar grid with the axis of rotation set to y-axis. 
    The observer is located at Z -> np.inf. 

    Args:
        active_cord (array): Cartesian coordinates of active regions (arr[x, y, z]) on the stellar grid.
        phase (integer): Rotational angle [in deg]

    Returns:
        [float, float, float]: Rotated corrdinates of active regions. 
    """
    rot= rotation_matrix(phase, 'y').T
    rotated_active_cord= np.dot(rot, active_cord)
    spx= rotated_active_cord[0][0]
    spy= rotated_active_cord[1][0]
    spz= rotated_active_cord[2][0]

    return spx, spy, spz