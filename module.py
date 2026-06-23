import matplotlib.pyplot as plt
import numpy as np
import emcee
from .misc import *
from scipy import interpolate
import json
import os
import subprocess
import sys
import corner
import pickle
import time


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ('1', 'true', 'yes', 'on')


class elmodel:
    def __init__(self, model, ini_file=None, x=None, y=None, nbands=1, logspace=None):
        '''
            Initialize the class. 
            Usage:
                loc = elmodel(model)
            Input:
                model = an array containing the model parameters R, v, rho, tgas
                x     = observed x-array
                y     = observed y-array
                nbands= number of bands in the observed spectra
                logspace = if some parameters should be in log-space, provide a Boolean list of the same length as the number of parameters.
        '''
        self.R, self.v, self.rho, self.tgas = model[:].T
        self.nbands = nbands
        self.V_obs = [[]]*nbands
        self.T_obs = [[]]*nbands
        self.ini_file = ini_file
        self.x = x
        self.y = y
        self.yerr = np.ones_like(y)
        self.logspace = logspace
        print("** -- elmod class initialized -- **")
        print("Model loaded, ini_file set", self.ini_file)
        print("Model expected to have %d bands" % self.nbands)
        print("-- ** !! ** --")
        print("Remember to set the target beam sizes, band_fnames, V_lsr, and load the observations")
        print("and remember to set the yerr (errors in the observed spectra).")
        print("Also, remember to set the prior function.")
        print("-- ** !! ** --")
        self._loc_worker = None

    # --------------- # --------------- # ---------------
    def _verbose(self):
        return _env_flag('ELMOD_VERBOSE')

    # --------------- # --------------- # ---------------
    def write_model(self, theta, N=51, model_fname='model.cloud'):
        '''
            Write LOC model file named: model.cloud
            Theta is list/array with the parameters needed for the specific model. 
            This needs to be updated to the specific model you are using!
        '''
        ### write model binary file:
        # allocate the cube C
        arr       = np.ones((N, 5), np.float32)   # start with values 1.0 for all parameters

        rho_scaling, ab = theta

        # rho, Tkin, sigma, abu, vrad
        arr[:,0] = self.rho * rho_scaling
        arr[:,1] = self.tgas
        arr[:,2] = 0.075 # km/s
        arr[:,3] = ab #np.power(10, ab)
        arr[:,4] = -self.v

        # Write the actual file
        fp = open(model_fname, 'wb')
        np.asarray([N], np.int32).tofile(fp)
        np.asarray([self.R/self.R.max()], np.float32).tofile(fp)
        arr.tofile(fp)
        fp.close()

    # --------------- # --------------- # ---------------
    def LOC(self, theta, rad_off=0):
        '''
            Run LOC. Updates the LOC model based on theta parameters.
            This file needs to be updated based on the number of spectra, the desired beam size etc.

        '''
        # Use the OS pid directly and avoid multiprocessing internals in LOC.
        pid = os.getpid()
        if self._verbose():
            print("Running LOC for PID %d with parameters: " % pid, theta, flush=True)
        ### model results:
        from LOC_aux import ReadIni
        ini_file = ReadIni(self.ini_file)
        dist = ini_file["distance"]
        angle = ini_file["angle"] #radius of the model in arcsec

        # convert theta from log-space if needed:
        if self.logspace is not None and (np.array(self.logspace) == True).any():
            theta = np.array(theta)
            theta[self.logspace] = 10**theta[self.logspace]


        # ---------------
        # write LOC model using parameters in theta:
        self.write_model(theta, model_fname='model_%d.cloud' % pid)

        # --------------- 
        # set the ini_file to use the model we just wrote:
        # create a copy of the ini_file with the correct model name:
        ini_new = 'loc_ini_%d.ini' % pid
        base = ini_file.get('prefix', 'prefix')
        with open(self.ini_file, 'r') as f:
            with open(ini_new, 'w') as f2:
                for line in f:
                    if line.startswith('cloud'):
                        f2.write('cloud          model_%d.cloud\n' % pid)
                    elif line.startswith('prefix'):
                        base = line.split()[1]
                        f2.write('prefix         %s_%d\n' % (base, pid))
                    else:
                        f2.write(line)
                    

        if self._verbose():
            print(self.ini_file, '-->', ini_new, flush=True)
        # --------------- # ---------------
        # check the files exist:
        if not os.path.isfile(ini_new):
            raise ValueError('ini_file %s does not exist' % ini_new)
        if not os.path.isfile('model_%d.cloud' % pid):
            raise ValueError('model file model_%d.cloud does not exist' % pid)
        


        self._run_loc1d(ini_new)

        # remove the temporary ini file
        #subprocess.call(['rm', ini_new])

        # --------------- # --------------- # --------------- 
        # Now we read the results and do convolution with e.g., 30m beam
        # read results and return:
        y_bands = []
        for i in range(self.nbands):
            targ_beam = self.targ_beams[i]
            filename = self.band_fnames[i]
            filename = base + '_%d.band%i.spe' % (pid, i)

            V, T =  convolve_loc_sps(filename, targ_beam, angle_as=None, samples=None)
            nray, _ = T.shape#number of rays, number of channels
            V = V + self.V_lsr

            # index for the spectra. This is always toward the core center in my case (single pointing)
            ind = np.int32(rad_off/dist/(angle/nray-1))
    
            # create interpolation functions.
            fint = interpolate.interp1d(V, T[ind, :], bounds_error=False, fill_value=(T[ind, np.argmin(V)], T[ind, np.argmax(V)]))
        
            # split observed x-arrays (if multiple bands)
            x_band = self.V_obs[i]
            # interp model onto observed x-arrays
            interp_y = fint(x_band)
            y_bands.append(interp_y)


        return np.concatenate(y_bands)

    # --------------- # --------------- # ---------------
    def _loc_env(self):
        env = os.environ.copy()
        env['PYTHONFAULTHANDLER'] = '1'
        env['PYTHONUNBUFFERED'] = '1'
        env['PYOPENCL_NO_CACHE'] = '1'
        return env

    # --------------- # --------------- # ---------------
    def _run_loc1d(self, ini_file):
        if os.environ.get('ELMOD_LOC_BACKEND', 'worker').lower() == 'subprocess':
            self._run_loc1d_subprocess(ini_file)
        else:
            self._run_loc1d_worker(ini_file)

    # --------------- # --------------- # ---------------
    def _run_loc1d_subprocess(self, ini_file):
        result = subprocess.run(
            [sys.executable, '-X', 'faulthandler', 'LOC1D.py', '%s' % ini_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=self._loc_env())
        if result.returncode != 0:
            if result.returncode < 0:
                reason = "signal %d" % (-result.returncode)
            else:
                reason = "exit code %d" % result.returncode
            raise RuntimeError(
                "LOC1D.py failed with %s:\n%s" %
                (reason, result.stdout))

    # --------------- # --------------- # ---------------
    def _ensure_loc_worker(self):
        if self._loc_worker is not None and self._loc_worker.poll() is None:
            return self._loc_worker
        self._loc_worker = subprocess.Popen(
            [sys.executable, '-u', 'loc_worker.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._loc_env())
        return self._loc_worker

    # --------------- # --------------- # ---------------
    def _run_loc1d_worker(self, ini_file):
        worker = self._ensure_loc_worker()
        request = json.dumps({"ini_file": ini_file})
        try:
            worker.stdin.write(request + "\n")
            worker.stdin.flush()
            line = worker.stdout.readline()
        except BrokenPipeError:
            line = ''
        if not line:
            stderr = worker.stderr.read()
            self._loc_worker = None
            raise RuntimeError("LOC worker stopped before responding:\n%s" % stderr)
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError("LOC worker failed:\n%s" % response.get("output", ""))

    # --------------- # --------------- # ---------------
    def load_obs(self, fname, band=0, header=0):
        ''' 
            Function to read the observed spectra from fname (containing V_obs, T_obs)
            Usage:
                loc.loadObs(fname, banad=0, header=0)
            Input:
                fname = name of the file containing the observed spectra
                band  = band number
                header = number of header lines in the file
        '''
        ### data:
        self.V_obs[band], self.T_obs[band] = np.genfromtxt(fname, skip_header=header).T
        self.x = np.concatenate(self.V_obs)
        self.y = np.concatenate(self.T_obs)

    # --------------- # --------------- # ---------------
    def pass_priorfunc(self, fprior):
        '''
            Pass the prior function to the class.
            Usage:
                loc.pass_priorfunc(func_prior)
            Input:
                fprior = prior function defined to the specific number of parameters in theta.
        '''
        self.fprior = fprior

    # --------------- # --------------- # ---------------
    def lnprior(self, theta):
        '''
            Prior function for the model.
            Usage:
                loc.lnprior(theta)
            Input:
                theta = model parameters
        '''
        return self.fprior(theta)

    # --------------- # --------------- # ---------------
    def _validated_model_and_errors(self, theta):
        y_model = np.asarray(self.LOC(theta), dtype=float)
        y_obs = np.asarray(self.y, dtype=float)
        yerr = np.asarray(self.yerr, dtype=float)

        if y_model.shape != y_obs.shape:
            return None, None, None
        if not np.all(np.isfinite(y_model)):
            return None, None, None
        if not np.all(np.isfinite(y_obs)):
            return None, None, None
        if not np.all(np.isfinite(yerr)) or np.any(yerr <= 0.0):
            return None, None, None
        return y_model, y_obs, yerr
    
    # --------------- # --------------- # ---------------
    def lnlike2(self, theta):
        '''
            Likelihood function for the model.
            Usage:
                loc.lnlike(theta)
            Input:
                theta = model parameters
        '''

        y_model, y_obs, yerr = self._validated_model_and_errors(theta)
        if y_model is None:
            return -np.inf
        inv_sigma2 = 1.0/(yerr**2)
        output = -0.5*(np.sum((y_obs-y_model)**2*inv_sigma2 - np.log(inv_sigma2)))
        if not np.isfinite(output):
            return -np.inf
        return output
    
    # --------------- # --------------- # ---------------
    def lnlike(self, theta):
        '''
            Likelihood function for the model.
            Usage:
                loc.lnlike(theta)
            Input:
                theta = model parameters
        '''
        y_model, y_obs, yerr = self._validated_model_and_errors(theta)
        if y_model is None:
            return -np.inf
        sigma2 = yerr**2 
        ## This is basically chi2:
        output = -0.5 * np.sum((y_obs - y_model) ** 2 / sigma2)
        if not np.isfinite(output):
            return -np.inf
        return output

    # --------------- # --------------- # ---------------
    def lnprop(self, theta):
        '''
            Proposal function for the model.
            Usage:
                loc.lnprop(theta)
            Input:
                theta = model parameters
        '''

        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        prop = lp + self.lnlike(theta)
        if not np.isfinite(prop):
            return -np.inf
        return prop
    
    # --------------- # --------------- # ---------------
    def check_loc(self):
        '''
            Check if the LOC properties are set.
        '''

        # check if self.x is defined
        if not hasattr(self, 'x'):
            raise ValueError('x is not defined. Please define x before running run_mcmc')
        # check if self.y is defined
        if not hasattr(self, 'y'):
            raise ValueError('y is not defined. Please define y before running run_mcmc')
        # check if self.yerr is defined
        if not hasattr(self, 'yerr'):
            raise ValueError('yerr is not defined. Please define yerr before running run_mcmc')
        # check if self.V_obs is defined
        if not hasattr(self, 'V_obs'):
            raise ValueError('V_obs is not defined. Please define V_obs before running run_mcmc')
        # check if self.T_obs is defined
        if not hasattr(self, 'T_obs'):
            raise ValueError('T_obs is not defined. Please define T_obs before running run_mcmc')
        # check if self.ini_file is defined:
        if not hasattr(self, 'ini_file'):
            raise ValueError('ini_file is not defined. Please define ini_file before running run_mcmc')
        # check if targ_beams is defined:
        if not hasattr(self, 'targ_beams'):
            raise ValueError('targ_beams is not defined. Please define targ_beams before running run_mcmc')
        # check if band_fnames is defined:
        if not hasattr(self, 'band_fnames'):
            raise ValueError('band_fnames is not defined. Please define band_fnames before running run_mcmc')
        # check if V_lsr is defined:
        if not hasattr(self, 'V_lsr'):
            raise ValueError('V_lsr is not defined. Please define V_lsr before running run_mcmc')
        # Check if the length of targ_beams is the same as the number of bands
        if len(self.targ_beams) != self.nbands:
            raise ValueError('The number of target beams should be the same as the number of bands')
        # Check if the length of band_fnames is the same as the number of bands
        if len(self.band_fnames) != self.nbands:
            raise ValueError('The number of band filenames should be the same as the number of bands')
        # Check if the length of V_obs is the same as the number of bands
        if len(self.V_obs) != self.nbands:
            raise ValueError('The number of V_obs should be the same as the number of bands')
        # Check if the length of T_obs is the same as the number of bands
        if len(self.T_obs) != self.nbands:
            raise ValueError('The number of T_obs should be the same as the number of bands')

    # --------------- # --------------- # ---------------
    def _validate_initial_walkers(self, pos):
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2:
            raise ValueError('pos must be a 2D array with shape (nwalkers, ndim)')
        if not np.all(np.isfinite(pos)):
            raise ValueError('Initial walker positions contain NaN or infinite values')

        log_prior = np.array([self.lnprior(theta) for theta in pos], dtype=float)
        bad = np.where(~np.isfinite(log_prior))[0]
        if len(bad) > 0:
            raise ValueError(
                'Initial walker positions outside prior at indices %s' %
                bad.tolist())
        return pos

        
    # --------------- # --------------- # ---------------
    def run_mcmc(self, pos, nsteps=1000, burnin=100, reset=True, fname='elmod_res.h5', progress=True):
        '''
            Run the MCMC for the model.
            Usage:
                loc.run_mcmc(nsteps=1000, burnin=100, reset=True)
            Input:
                pos = starting position for the walkers
                nsteps   = number of steps
                burnin   = burnin steps
                reset    = reset the walkers (Default: True)
                fname    = name of the HDF5 file to save the walker chain (Default: elmod_res.h5)
                progress = show emcee progress bars (Default: True)
            Note:
                The results are saved in the file elmod_res.h5 or the name you provide in fname.
        '''
        # check that the LOC properties are set:
        self.check_loc()
        pos = self._validate_initial_walkers(pos)

        print("Running MCMC with nwalkers = %d, nsteps = %d, burnin = %d" % (pos.shape[0], nsteps, burnin))
        # number of walkers and dimensions
        nwalkers, ndim = pos.shape

        ##### To save the walker chain I use HDF5. This is helpful if you want to restart a run on the cluster or your computer.
        backend = emcee.backends.HDFBackend(fname)
        if reset:
            ##### In this case we start from scratch (reset the walkers)
            backend.reset(nwalkers, ndim)

        ##### This is the MCMC (emcee) magic. 
        # We call the sampler. Tell it have many walkers and dimensions we have. Gives it our probability function and the observed data.
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.lnprop, backend=backend)
        # If burnin is not zero, run the burnin steps:
        state = pos
        if burnin > 0:
            print("Running burn-in...")
            state = sampler.run_mcmc(pos, burnin, progress=progress)
            sampler.reset()

        # Call the sampler to run the MCMC. The call here specifies the starting position pos and the number of steps for each walker.
        print("Running production...")
        sampler.run_mcmc(state, nsteps, progress=progress)

    # --------------- # --------------- # ---------------
    def run_mcmc_gpu(self, pos, nsteps=1000, burnin=100, reset=True, fname='elmod_res.h5', progress=True):
        '''
            Run the MCMC for the model using the GPU version of emcee.
            Usage:
                loc.run_mcmc_gpu(nsteps=1000, burnin=100, reset=True)
            Input:
                pos = starting position for the walkers
                nsteps   = number of steps
                burnin   = burnin steps
                reset    = reset the walkers (Default: True)
                fname    = name of the HDF5 file to save the walker chain (Default: elmod_res.h5)
                progress = show emcee progress bars (Default: True)
            Note:
                The results are saved in the file elmod_res.h5 or the name you provide in fname.
        '''
        # check that the LOC properties are set:
        self.check_loc()
        pos = self._validate_initial_walkers(pos)

        print("Running MCMC with nwalkers = %d, nsteps = %d, burnin = %d" % (pos.shape[0], nsteps, burnin))
        # number of walkers and dimensions
        nwalkers, ndim = pos.shape

        ##### To save the walker chain I use HDF5. This is helpful if you want to restart a run on the cluster or your computer.
        backend = emcee.backends.HDFBackend(fname)
        if reset:
            ##### In this case we start from scratch (reset the walkers)
            backend.reset(nwalkers, ndim)
        ##### setup a pool for the GPU
        from multiprocessing import Pool
        processes = min(nwalkers, os.cpu_count() or 1)
        with Pool(processes=processes) as pool:
            ##### This is the MCMC (emcee) magic. 
            # We call the sampler. Tell it have many walkers and dimensions we have. Gives it our probability function and the observed data.
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprop, backend=backend, pool=pool)
            # If burnin is not zero, run the burnin steps:
            state = pos
            if burnin > 0:
                print("Running burn-in...")
                state = sampler.run_mcmc(pos, burnin, progress=progress)
                sampler.reset()

            # Call the sampler to run the MCMC. The call here specifies the starting position pos and the number of steps for each walker.
            print("Running production...")
            sampler.run_mcmc(state, nsteps, progress=progress)


    # --------------- # --------------- # ---------------
    def plot_median(self, figname='median.png', filename='elmod_res.h5', \
    			thin=1, discard=50, offset_factor=0.5, plot_residual=True):
        '''
            Plot the median of the MCMC results.
            Usage:
                loc.plot_median(filename='elmod_res.h5', thin=2, discard=100)
            Input:
                figname  = name of the figure file
                filename = name of the HDF5 file containing the walker chain
                thin     = thinning factor for the walker chain
                discard  = number of steps to discard
        '''
        # Load the walker chain
        reader = emcee.backends.HDFBackend(filename)
        samples = reader.get_chain(discard=discard, flat=True, thin=thin)

        # Plot the median
        medians = np.median(samples, axis=0)
        print("Median values:   ", medians, flush=True)
        print("Calling LOC from plot_median", flush=True)
        y_model = self.LOC(medians)

        # ---------------------------------
        # create plot with panels for each band
        fig = plt.figure(figsize=(14, 10))
        axes = fig.subplots(self.nbands, 1, sharex=False, sharey=False)
        # perform split of y_model and self.x based on size of the bands
        for i in range(self.nbands):
            size_prev = len(np.concatenate(self.V_obs[0:max(i,1)]))
            if i == 0: size_prev = 0
            size_cur = len(self.V_obs[i])
            
            y = y_model[size_prev:size_prev+size_cur]

            axes[i].plot(self.V_obs[i], y, label='Model', lw=2.0)
            axes[i].plot(self.V_obs[i], self.T_obs[i], drawstyle='steps-mid', label='Observed', lw=2.0)
            if plot_residual:
                y_max = np.max(np.concatenate((y, self.T_obs[i])))
                y_min = np.min(np.concatenate((y, self.T_obs[i])))
                offset = y_min - y_max*offset_factor # offset for the residual plot
                # create residual plot:
                axes[i].plot(self.V_obs[i], (y - self.T_obs[i]) + offset, drawstyle='steps-mid', lw=2.0, label='Residual')
                axes[i].plot(self.V_obs[i], offset + np.zeros_like(self.V_obs[i]), '-', drawstyle='steps-mid', lw=1.5, c='k')
        fig.legend()
        fig.savefig(figname)
        plt.close()

    # --------------- # --------------- # ---------------
    def plot_highest_prob(self, figname='highest_prop.png', filename='elmod_res.h5', \
    			thin=2, discard=10, offset_factor=0.5, plot_residual=True):
        '''
            Plot the result with the highest probability.
            Usage:
                loc.plot_highest_prop(filename='elmod_res.h5', thin=2, discard=100)
            Input:
                figname  = name of the figure file
                filename = name of the HDF5 file containing the walker chain
                thin     = thinning factor for the walker chain
                discard  = number of steps to discard
        '''
        # Load the walker chain
        reader = emcee.backends.HDFBackend(filename)
        samples = reader.get_chain(discard=discard, flat=True, thin=thin)

        # Find the highest probability
        ind = np.argmax(reader.get_log_prob(discard=discard, flat=True, thin=thin))
        print("Highest probability values:   ", samples[ind], flush=True)
        print("Calling LOC from plot_highest_prob", flush=True)
        y_model = self.LOC(samples[ind])


        # ---------------------------------
        # create plot with panels for each band
        fig_hp = plt.figure(figsize=(14, 10))
        axes = fig_hp.subplots(self.nbands, 1, sharex=False, sharey=False)
        # perform split of y_model and self.x based on size of the bands
        for i in range(self.nbands):
            size_prev = len(np.concatenate(self.V_obs[0:max(i,1)]))
            if i == 0: size_prev = 0
            size_cur = len(self.V_obs[i])
            
            y = y_model[size_prev:size_prev+size_cur]

            axes[i].plot(self.V_obs[i], y, label='Model', lw=2.0)
            axes[i].plot(self.V_obs[i], self.T_obs[i], drawstyle='steps-mid', label='Observed', lw=2.0)
            if plot_residual:
                y_max = np.max(np.concatenate((y, self.T_obs[i])))
                y_min = np.min(np.concatenate((y, self.T_obs[i])))
                offset = y_min - y_max*offset_factor # offset for the residual plot
                # create residual plot:
                axes[i].plot(self.V_obs[i], (y - self.T_obs[i]) + offset, drawstyle='steps-mid', lw=2.0, label='Residual')
                axes[i].plot(self.V_obs[i], offset + np.zeros_like(self.V_obs[i]), '-', drawstyle='steps-mid', lw=1.5, c='k')
        fig_hp.legend()
        fig_hp.savefig(figname)
        plt.close()
        
    # --------------- # --------------- # ---------------
    def plot_corner(self, figname='corner.png', filename='elmod_res.h5', thin=2, discard=10):
        '''
            Plot the corner plot for the MCMC results.
            Usage:
                loc.plot_corner(filename='elmod_res.h5', thin=2)
            Input:
                figname  = name of the figure file
                filename = name of the HDF5 file containing the walker chain
                thin     = thinning factor for the walker chain
        '''
        # Load the walker chain
        reader = emcee.backends.HDFBackend(filename)
        samples = reader.get_chain(discard=discard, flat=True, thin=thin)
        truths = np.median(samples, axis=0)
        fig_corner = corner.corner(samples, labels=[r"$\rho$", r"$\log_{10}(\mathrm{abu})$"], truths=truths)
        plt.savefig(figname)
        plt.close()


    # --------------- # --------------- # ---------------
    def save(self, fname='elmod.pkl'):
        '''
            Save the class instance to a pickle file.
            Usage:
                loc.save(fname="elmod.pkl")
            Input:
                fname = name of the pickle file
        '''
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    # --------------- # --------------- # ---------------
    def load(self, fname='elmod.pkl'):
        '''
            Load the class instance from a pickle file.
            Usage:
                loc.load(fname="elmod.pkl")
            Input:
                fname = name of the pickle file
        '''
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        return obj
