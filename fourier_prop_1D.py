'''Fourier Propagation Software

This code contains a series of definitions to perform Fourier beam propagation around the focal plane of linearly polarized light.

Required Packages:
    numpy
    scipy
    matplotlib
    tqdm

'''

#Physical constants
c = 2.99792458e10 # cm/s^2

from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import numpy as np
import matplotlib as mpl
import tqdm




def draw_gaussian(x, x0, delta, super_gauss=1):
    '''Makes gaussian based on input array
    
    Parameters
    ----------
    name (data type) [physical unit]
    
    x (array) []: the independent variable
    x0 (float) []: center value
    delta (float) [arb. units]: 1/exp(2) radius of gaussian of the intensity distribution
    super_gauss (float) []: power for higher order gaussian
    
    Returns
    -------
    omegas (array) []: np.abs(np.exp(-(x - x0)**2 / delta**2)**super_gauss)
    '''

    return np.exp(-((x - x0)**2 / delta**2)**super_gauss)


def fraunhofer(u1, omegas, focus, dx1, N, wvl_0, interp=True, interp_kind='cubic'):
    '''Fraunhofer propagation
    
    Parameters
    ----------
    name (data type) [physical unit]
    
    u1 (array, complex64) []: the intial field defined in space-frequency
    omegas (array) [rad/s]: the frequency array of the field
    focus (float) [cm]: focal length of ideal lens
    dx1 (float) [cm]: grid spacing of the intitial simulation spatial window
    L (float) [cm]: length of initial simulation window
    N (float) []: number of spatial grid points in simulation window
    wvl_0 (float) [cm]: center wavlength of the spectrum
    interp (bool) []: sets whether or not the different scaled focal planes of each frequency is interpolated to the center frequency (to be correct this needs to be true)
    interp_kind () []: sets the interpolation method
    
    Returns
    -------
    u2 (array) []: the field at the focus in space-frequency 
    eFieldxt (array) []: the field at the focus in space-time
    x2 (array) [cm]: the x-dimension of the focal plane (of the center frequency)
    dx2 (float) [cm]: the grid spacing in the x-dimension of the focal plane window
    '''

    #Initialize

    N_omega = len(omegas)
    i = 0

    u2 = np.zeros((N_omega, N), dtype=np.complex64)

    #length of rescaled simulation window at focus for center wavelength
    L2_interp = wvl_0 * focus /dx1

    #rescaled x array of the new simulation window at focus
    xnew = np.linspace(-1 * L2_interp / 2, L2_interp / 2, N)
    
    
    for omega in tqdm.tqdm(omegas):

        wvl, k = get_wavelength(omega)

        #fraunhofer propagation

        #length of rescaled simulation window at focus for current wavelength in loop
        L2 =wvl * focus / dx1

        #define new spatial array for current wavenlength in loop
        x2 = np.linspace(-L2 / 2, L2 / 2, N)

        #fraunhofer propagator
        cc = 1 / (1j * wvl * focus) * np.exp(1j * k / (2 * focus) * (x2**2))
        u2[i] = cc * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(u1[i]))) * dx1
       
        if interp:
            #rescales each frequency's focal plane to the same dimensions as the center frequency
            f_real = interpolate.interp1d(x2, u2[i].real, kind=interp_kind,fill_value="extrapolate") # WARNING: significant interpolation errors will occur at sharp jumps in E-field with cubic
            f_imag = interpolate.interp1d(x2, u2[i].imag, kind=interp_kind,fill_value="extrapolate")
            
            u2[i] = f_real(xnew) + 1j*f_imag(xnew)

        i+=1
    plt.show()

    #Return only x values and grid spacing at focus
    x2 = xnew
    dx2 = np.abs(x2[0] - x2[1])
    eFieldxt = ifft(u2)

    return u2, eFieldxt, x2, dx2


def fwhm(ind_var, dep_var, percent_max=0.5, verbose=False):
    '''Find full width at specified percentage of maximum of data
    
    Parameters
    ----------
    name (data type) [physical unit]
    
    dep_var (array) []: the data to retrieve FWHM from
    ind_var (array) []: the x-axis values
    percent_max (float) []: (default = 0.5) percentage of maximum to evaluate full width at
    verbose (bool) []: if true will print the FWHM at percent max value
    
    Returns
    -------
    fwhm (float) [ind_var units]: FWHM of data in the units of the independent variable x values
    '''

    spline = UnivariateSpline(ind_var, dep_var - np.max(dep_var) * percent_max, s=0)
    if np.size(spline.roots()) == 0:
        r1 = 0
        r2 = 0
    else:
        r1 = spline.roots()[0] # find the roots
        r2 = spline.roots()[-1]
    fwhm = np.abs(r1 - r2)
    
    if verbose:
        print('Full width @ ', percent_max, '* maximum = ', fwhm)
        
    return fwhm


def get_wavelength(omega):
    '''Calculates wavelength and wavenumber from frequency
    
    Parameters
    ----------
    name (data type) [physical unit]
    
    omega (array or float) [rad/s]: the frequency to convert
    
    Returns
    -------
    wvl (float) [cm]: the converted wavelength
    k (float) [1/cm]: the converted wavenumber
    '''

    wvl = 2 * np.pi * c / omega
    k = omega / c
    return wvl, k


def ifft(u, axis=0):
    '''
    Does the inverse fourier transform with appropriate fftshift
    
    Parameters
    ----------
    name (data type) [physical unit]
    
    u (array, complex64) []: input field to perform calculation on
    
    Returns
    -------
    (array, complex64) []: fourier transformed field
    '''

    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(u), axis=axis))


def plot_gaussian_spectrum(omegas,
                           omega_0,
                           delta_omega,
                           percent_max=0.5,
                           print_full_width=True,
                           show_center_freq=False,
                           show_delta_width=False,
                           super_gauss=1,
                           xlabel='ω',
                           ylabel='Intensity, [arb. units]'):
    '''Assumes gaussian spectrum and shows a plot
    
    Parameters
    ----------
    name (data type) [physical unit]
    
    omegas (array) []: frequency array
    omega_0 (float) []: center frequency
    delta_omega (float) []: 1/exp(2) radius of the spectrum of the field intensity
    percent_max (float) []: percentage of maximum to evaluate full width at
    print_full_width (bool) []: if true prints out full width value at percent_max
    show_center_freq (bool) []: if true plot vertical line at center frequency
    show_delta_width (bool) []: if true plot vertical lines at the + and - delta_omega values from the center frequency
    super_gauss (float) []: factor to make a higher order gaussian
    xlabel (string) []: label of the x axis
    ylabel (string) []: label of the y axis
    
    Returns
    -------
    () []: 
    '''

    y = draw_gaussian(omegas, omega_0, delta_omega, super_gauss=super_gauss)
    
    if print_full_width:
        width = fwhm(y, omegas, percent_max=percent_max, verbose=True)
    

    plt.figure()
    plt.plot(omegas, y)
    if show_delta_width:
        plt.axvline(x=omega_0 - delta_omega)
        plt.axvline(x=omega_0 + delta_omega)
    if show_center_freq:
        plt.axvline(x=omega_0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
    
def aspw(u, x, dx, omegas, zs, timeshift, cmap=None, plot=True, savefig=None):
    '''Angular Spectrum of plane waves (ASPW) propagation
    
    Parameters
    ----------
    name (data type) [physical unit]

    u (array) []: field to be propagated
    x (array) [cm]: x array of the field
    omegas (array) [rad/s]: frequency array of the field
    zs (array) [cm]: the z values to propagate to from the focus
    cmap (cmap) []: the colormap to use for plotting, defaults to custom colormap
    plot (bool) []: if true plots the space time field intensity at the set z position
    savefig (string) []: path to save folder
    
    Returns
    -------
    intensity () []: returns the intensity as a function of z
    eFieldxt2 (array) []: return the field at the focuse in (x,t)
    '''

    # Initialize
    N = len(x)
    N_omega = len(omegas)
    N0 = int(N / 2) # used to return central intensity
    j = 0
    single_zs_flag = False

    kx = np.array(np.fft.fftshift(2*np.pi*np.fft.fftfreq(N, dx)), dtype=np.float64) # Spatial frequencies
    Kx = kx
    intensity = []
    
    if not zs.shape: # if only 1 z, then use tqdm on omegas instead
        zs = np.array([zs])
        
        single_zs_flag = True
    elif len(zs) == 1:
        
        single_zs_flag = True
    else:
        zs = tqdm.tqdm(zs)
    
    for z in zs:
        
        i = 0
        eFieldxw1 = np.zeros((N_omega, N), dtype=np.complex64)

        for omega in omegas:

            wvl, k = get_wavelength(omega)

            Fk = np.array(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(u[i]))), dtype=np.complex64)

            kz = k - 0.5*(Kx**2) / k

            #Propagator
            Hk = np.array(np.exp(1j * kz * z) * np.exp(-1j * k * z), dtype=np.complex64)
            #Final spatial frequency function 
            Gk = np.multiply(Fk, Hk) 
            
            #Output spatial function as the inverse FFT of Gk
            eFieldxw1[i] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Gk)))

            i+=1

        eFieldxt2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(eFieldxw1), axis=0))

        if savefig is not None or plot:
            
            fig_data = np.abs(eFieldxt2[:,:].T**2)
            

            plt.imshow(fig_data[:,::-1], aspect='auto', origin='lower', extent=[timeshift.min()*1e15, timeshift.max()*1e15, x.min(), x.max()], cmap=cmap)
            plt.xlabel('time [fs]')
            plt.ylabel('x [cm]')
            plt.title("z = % .4f" % z)

            if savefig is not None:
                plt.savefig(savefig+"/frame"+str(j).zfill(2)+".png", bbox_inches="tight")
            if plot:
                plt.show()

        intensity.append(np.max(np.abs(eFieldxt2[:,N0].T**2)))

        
        if single_zs_flag:
            return intensity, eFieldxt2
        
        j+=1
    
    return intensity


def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python or
    https://gist.github.com/error454/65d7f392e1acd4a782fc
    This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range

    Parameters
    ----------
    name (data type) [physical unit]

    wavlength (float) [nm]: the wavelngth to convert to an rgb value

    Returns
    -------
    (R,G,B,A) () []: returns color value of a wavelength 
    '''
    
    wavelength = float(wavelength-200)#subtract 200 so NIR light is converted to VIS for visualization
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)


def tiltAngle(eFieldxt,t_bound,x_bound,PLOTS=False):
    '''calculate the angle of tilt from the z-axis of the beam by fitting the beam profile with a 2D gaussian
    fit routine courtesy of Christian Hill (https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/#rating-111)
    Parameters
    ----------
    name (data type) [physical unit]

    eFieldxt (complex) []: the input E(x,t) to calculate the tilt of
    t_bound (float) [fs]: the positive bound of the time axis of the field
    x_bound (float) [um]: the positive bound of the x axis of the field
    PLOTS (bool) []: if True, will plot the beam profile and a countour plot overlay of the fit (generated from the fit parameters in popt)

    Returns
    -------
    fit_angle (float) [degrees]: the angle determined by the fit, counter-clockwise from the positive z-axis
    '''


    eFieldxt_center_subset=eFieldxt
    
    #normalize and multiply to a factor to ensure a good fit (amplitude needs to be sufficiently large)
    Z=np.abs(eFieldxt_center_subset)**2/np.max(np.abs(eFieldxt_center_subset)**2)*9
    
    
    x1, y = np.linspace(-t_bound*0.3,t_bound*0.3,np.shape(eFieldxt)[1]), np.linspace(-x_bound,x_bound,np.shape(eFieldxt)[0])
    X, Y = np.meshgrid(x1, y)
    
    def gaussian(x, y, A, x0, y0, sigmax, sigmay, theta):
        a = ((np.cos(theta)**2) / (2*sigmax**2)) + ((np.sin(theta)**2) / (2*sigmay**2))
        b = -((np.sin(2*theta)) / (4*sigmax**2)) + ((np.sin(2*theta)) / (4*sigmay**2))
        c = ((np.sin(theta)**2) / (2*sigmax**2)) + ((np.cos(theta)**2) / (2*sigmay**2))
        return A*np.exp(-(a*(x-x0)**2+c*(y-y0)**2 + 2*b*(x-x0)*(y-y0)))
    
    def _gaussian(M, *args):
        x, y = M
        arr = gaussian(x, y, *args)
        return arr

    # Initial guesses to the fit parameters.
    # A list of the Gaussian parameters: A, x0, y0, sigma_x, sigma_y, theta
    guess_prms = [(2, 0, 0, 10, 10, np.deg2rad(45))]#,

    # Flatten the initial guess parameter list.
    p0 = [p for prms in guess_prms for p in prms]

    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))
    # Do the fit, using our custom _gaussian function which understands our
    # flattened (ravelled) ordering of the data points.
    popt, pcov = curve_fit(_gaussian, xdata, Z.ravel(), p0)

    fit = gaussian(X,Y, *popt)
    if PLOTS==True:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.imshow(Z, origin='lower',
                  extent=(x1.min(), x1.max(), y.min(), y.max()),aspect='auto')
        ax.contour(X, Y, fit, colors='w',levels=[1,3],origin = 'lower')
        ax.set_xlabel('z [um]')
        ax.set_ylabel('x [um]')
        plt.show()
    
    fit_angle = 90-(np.rad2deg(popt[5]) % 90) 
    return fit_angle



class FourierProp:
    
    #Define physical constants
    
    c = 2.99792458e10 # cm
    e0 = 8.8541878128e-18 # units
    eps_c = (8.8541878128e-14 * c / 2)
    s2fs = 1e15
    
    
    def __init__(self,
                 approximation=False,
                 BAR=1,
                 chirp_pattern='radial',
                 cmap=None,
                 delta_wvl=100e-7,
                 deltax=0.68494003,
                 focus=5,
                 gauss_power=1,
                 L=5,
                 L_omega=None,
                 N=2**9 + 1,
                 N_omega=2**8 + 1,
                 w_in=0.05,
                 wvl_0=800e-7,
                 separation=0.5,
                 d=1/(1000*10),
                 aoi=0,
                 m=1):
        
        # --- Set default values
        self.approximation = approximation #approximation that wvl,k = wvl_0,k_0 in propagtor
        self.BAR = BAR #beam aspect ratio fo the chirped beam. BAR=1 is unchirped gaussian, BAR=2 is chirped dimension diamter is twice the unchirped diamter, etc.
        self.chirp_pattern = chirp_pattern #set the chirp pattern to simulate
        self.cmap = cmap #colormap to use for plotting
        self.delta_wvl = delta_wvl #FWHM of spectrum
        self.deltax = deltax #shift of center frequency from z-axis
        self.focus = focus #focal length to propagate to
        self.gauss_power = gauss_power #can set higher order gaussian
        self.L = L #length os simulation window along x
        self.L_omega = L_omega #length of simulation window along frequency
        self.N = N #number of grid points in the spatial dimension
        self.N_omega = N_omega #number of grid points in the frequency dimension
        self.w_in = w_in #input 1/exp(2) beam radius of the field intensity
        self.wvl_0 = wvl_0 #center wavelength of spectrum
        self.separation = separation #separation between gratings
        self.d = d #groove density of gratings 
        self.aoi = aoi #angle of incidence on grating
        self.m = m #diffraction order of grating
        
        self.initialize(verbose=False)
        
        
    def construct_eField(self, plot=False):
        '''Constructs default e-field for various chirp patterns

        Parameters
        ----------
        name (data type) [physical unit]

        chirp_pattern (string): value setting what type of chirp pattern is to be simulated, can be linear, linear_grating, linear_grating_centered
            linear: sets chirp to linear approximation around the center frequency
            linear_grating: calculates chirp from grating pair. Pulse is offset from the z-axis from the diffraction of the frequencys from normal
            linear_grating_centered: centers the chirp from a grating pair so that the center frequency is on axis at x=0

        Returns
        -------
        () []: 
        '''

        i = 0
        #initialize
        alpha = self.alpha
        c = self.c
        chirp_pattern = self.chirp_pattern
        delta_omega = self.delta_omega
        deltax = self.deltax
        gauss_factor = self.gauss_power
        L = self.L
        N_omega = self.N_omega
        N = self.N
        omega_0 = self.omega_0
        omegas = self.omegas
        w_in = self.w_in
        x = self.x
        aoi=self.aoi
        d=self.d
        separation=self.separation
        m=self.m

        #define the position along x of each diffracted frequency. If it is an invalid solution to the gratign equation, set to 1000 which will b processed later in the code
        x_chirp=np.nan_to_num(separation*np.tan(np.arcsin(m*2*np.pi*c/(omegas*d)-np.sin(aoi))),nan=1000)

        X = x
        self.u1 = np.zeros((N_omega, N), dtype=np.complex64)
        self.u2 = np.zeros((N_omega, N), dtype=np.complex64)
        self.x_vals=np.zeros((N_omega))

        if plot:
            lnst = '-'
            alph = 1
            plt.figure(figsize=(8, 6))
            plt.xlabel('x [cm]')
            plt.ylabel('Intensity [arb. units]')

        # TODO: get rid of loop, this can be done without looping
        for omega in tqdm.tqdm(omegas):

            if chirp_pattern == 'linear':
                u1_spatial = np.array(np.exp(-1*((((X+deltax)-alpha*(omega-omega_0))/(w_in))**2)**gauss_factor), dtype=np.complex64)#-np.abs(alpha*(omega-omega_0))
                self.x_vals[i]=alpha*(omega-omega_0)

            elif chirp_pattern == 'linear_grating':
                u1_spatial = np.array(np.exp(-1*((((X)-x_chirp[i])/(w_in))**2+(Y/w_in)**2)**gauss_factor), dtype=np.complex64)
                
            elif chirp_pattern == 'linear_grating_centered':
                u1_spatial = np.array(np.exp(-1*((((X)+x_chirp[i]-(separation*np.tan(np.arcsin(m*2*np.pi*c/(omega_0*d)-np.sin(aoi)))))/(w_in))**2)**gauss_factor), dtype=np.complex64)
                #sign inverted to put blue on positive x-axis
                self.x_vals[i]=np.nan_to_num(separation*np.tan(np.arcsin(-1*2*np.pi*c/(omega*d)+np.sin(aoi))),nan=1000)
                if x_chirp[i]==1000:
                    u1_spatial=np.zeros((np.shape(u1_spatial)))

            else: error('chirp_pattern not recognized')

            u1_freq = np.exp(-((omega - omega_0)**2 / delta_omega**2)**gauss_factor)
            u1 = u1_freq * u1_spatial
            self.u1[i] = u1

            if plot:
                plt.plot(x, np.abs(u1**2), color=wavelength_to_rgb(2*np.pi*c/omega*1e7), linestyle=lnst, alpha=alph)

            i+=1
            
        if plot:
            plt.show()
                    
            
    def fraunhofer(self):
        self.u2, self.eFieldxt, self.x2, self.dx2 = fraunhofer(self.u1, self.omegas, self.focus, self.dx1, self.N, self.wvl_0)
        
        
    def initialize(self, verbose=True):
        '''Initializes and prepares input data for propagation
        
        Must be run each time a parameter gets changed
        
        Parameters
        ----------
        name (data type) [physical unit]

        verbose (bool) []: (default = True)

        Returns
        -------
        none
        '''

        # --- Spatial domain
        self.bb = np.sqrt(self.BAR**2 - 1)
        self.L=self.L
        self.x = np.linspace(-self.L / 2, self.L / 2, self.N)
        self.dx1 = np.abs(self.x[0] - self.x[1])
        
        # --- Frequency domain
        self.omega_0 = 2 * np.pi * self.c / self.wvl_0
        self.delta_omega = 2 * np.pi * self.c * self.delta_wvl / (np.sqrt(2 * np.log(2)) * self.wvl_0**2)
        if self.L_omega is None: self.L_omega = 4*2 * 8 * self.delta_omega
        self.omegas = np.linspace(1, self.L_omega, self.N_omega)
        if self.N_omega == 0:
            self.N_omega=2**13+1
            self.omegas = np.linspace(self.omega_0 - self.L_omega/2, self.omega_0 + self.L_omega/2, self.N_omega)
        self.d_omega = np.abs(self.omegas[0] - self.omegas[1])

        self.alpha = self.w_in * self.bb / self.delta_omega
        
        # --- Time domain
        self.time = np.fft.fftfreq(self.N_omega, self.d_omega / (2 * np.pi))
        self.timeshift = np.fft.fftshift(self.time)
        self.dt = np.abs(self.timeshift[0] - self.timeshift[1])
        
        # --- Colormap
        if self.cmap is None: self.set_default_colormap()
        
        if verbose:
            print('    L_x:', self.L)
            print('    d_x:', self.dx1)
            print('    N_x:', self.N)
            print('-----------------------------')
            print('L_omega:', self.L_omega)
            print('d_omega:', self.d_omega)
            print('N_omega:', self.N_omega)
            print('    d_t:', self.dt)
            print('-----------------------------')
            print('    BAR:', self.BAR)
            print('    delta_omega:', self.delta_omega)

                  
    def set_default_colormap(self):
        '''generates a custom colormap from local .npy file

        Parameters
        ----------
        
        Returns
        -------
        
        '''
            
        #Load colormap from file
        cmap = np.load("./colormap.npy")

        #Convert to matplotlib colormap
        self.cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
        
        
    def aspw(self, zs=np.linspace(-0.05,1,1), plot=True, savefig=None):
        if not zs.shape or len(zs) == 1:
            self.intensity, self.eFieldxt2 = aspw(self.u2, self.x2, self.dx2, self.omegas, zs, self.timeshift, cmap=self.cmap, plot=plot, savefig=savefig)
        else:
            self.intensity = aspw(self.u2, self.x2, self.dx2, self.omegas, zs, self.timeshift, cmap=self.cmap, plot=plot, savefig=savefig)
