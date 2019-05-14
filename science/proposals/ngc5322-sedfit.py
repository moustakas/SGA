#!/usr/bin/env python
"""
Fit the SED of NGC5322 using Prospector.

"""
import os, time, argparse, pdb
import numpy as np
import multiprocessing

ang2micron = 1e-4 # Angstrom --> micron
maggies2mJy = 10**(0.4*16.4) # maggies --> mJy

def _niceparnames(parnames):
    """Replace parameter names with nice names."""

    old = list(['tau',
           'tage',
           'mass',
           'logmass',
           'logzsol',
           'dust2'])
    new = list([r'$\tau$ (Gyr)',
           'Age (Gyr)',
           r'$M / M_{\odot}$',
           r'$\log_{10}\,(M / M_{\odot})$',
           r'$\log_{10}\, (Z / Z_{\odot})$',
           r'$\tau_{diffuse}$'])

    niceparnames = list(parnames).copy()
    for oo, nn in zip( old, new ):
        this = np.where(np.in1d(parnames, oo))[0]
        if len(this) > 0:
            niceparnames[this[0]] = nn
            
    return np.array(niceparnames)

def _galaxyphot(obs):
    """Get the galaxy photometry and inverse variances (converted to mJy) and filter
    effective wavelengths (converted to microns).

    """
    weff = np.array([f.wave_effective for f in obs['filters']]) * ang2micron
    fwhm = np.array([f.effective_width for f in obs['filters']]) * ang2micron

    if False:
        galphot = obs['maggies'] * maggies2mJy
        galphoterr = obs['maggies_unc'] * maggies2mJy
    else:
        galphot = -2.5 * np.log10(obs['maggies'])
        galphoterr = 2.5 * obs['maggies_unc'] / obs['maggies'] / np.log(10)

    return weff, fwhm, galphot, galphoterr

def _sed(model, theta, obs, sps):
    """Construct the SED for a given set of parameters.  Divide by mextra to account
    for the *current* mass in stars (rather than the integrated stellar mass
    based on the SFH.

    Also convert wavelengths from Angstroms to microns and fluxes from maggies
    to mJy.

    """
    modelwave = sps.wavelengths * (1 + obs['redshift']) # [observed-frame wavelengths]
    modelwave *= ang2micron
    
    modelspec, modelphot, mextra = model.mean_model(theta, obs, sps=sps)
    if False:
        modelspec *= maggies2mJy
        modelphot *= maggies2mJy
    else:
        modelspec = -2.5 * np.log10(modelspec)
        modelphot = -2.5 * np.log10(modelphot)
    #print(modelphot)
    
    return modelwave, modelspec, modelphot

def bestfit_sed(obs, chain=None, lnprobability=None, theta=None, sps=None,
                model=None, seed=None, nrand=100, png=None):
    """Plot the (photometric) best-fitting SED.

    Either pass chain and lnprobability (to visualize the emcee fitting results)
    *or* theta (to visualize just a single SED fit).

    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, ScalarFormatter, FuncFormatter
    
    import seaborn as sns
    sns.set(style='ticks', font_scale=1.6, palette='Set2')
    
    rand = np.random.RandomState(seed)

    # Get the galaxy photometry and filter info.
    weff, fwhm, galphot, galphoterr = _galaxyphot(obs)

    # Build the maximum likelihood model fit and also grab a random sampling of
    # the chains with weight equal to the posterior probability.    
    if chain is not None:
        if chain.ndim == 3: # emcee
            nwalkers, niter, nparams = chain.shape
            ntot = nwalkers * niter
            flatchain = chain.reshape(ntot, nparams)
            lnp = lnprobability.reshape(ntot)
        else: # dynesty
            ntot, nparams = chain.shape
            flatchain = chain
            lnp = lnprobability
            
        theta = flatchain[lnp.argmax(), :] # maximum likelihood values
        print('Maximum likelihood values: ', theta)

        prob = np.exp(lnp - lnp.max())
        prob /= prob.sum()
        rand_indx = rand.choice(ntot, size=nrand, replace=False, p=prob)
        theta_rand = flatchain[rand_indx, :]

    print('Rendering the maximum-likelihood model...', end='')
    t0 = time.time()
    modelwave, modelspec, modelphot = _sed(model=model, theta=theta, obs=obs, sps=sps)
    print('...took {:.2f} sec'.format(time.time()-t0))
    #print(modelspec.min(), modelspec.max())

    # Establish the wavelength and flux limits.
    minwave, maxwave = 0.1, 1000.0
    #minwave, maxwave = np.min(weff - 5*fwhm), np.max(weff + fwhm)

    inrange = (modelwave > minwave) * (modelwave < maxwave)
    #maxflux = np.hstack( (galphot + 5*galphoterr, modelspec[inrange]) ).max() * 1.2
    #minflux = -0.05 * maxflux
    minflux, maxflux = (9, 24)

    fig, ax = plt.subplots(figsize=(8, 6))
    if chain is not None and nrand > 0:
        for ii in range(nrand):
            _, r_modelspec, _ = _sed(model=model, theta=theta_rand[ii, :], obs=obs, sps=sps)
            ax.plot(modelwave, r_modelspec, alpha=0.8, color='gray')
    ax.plot(modelwave, modelspec, alpha=1.0, label='Model spectrum', color='k')
    
    ax.errorbar(weff, modelphot, marker='s', ls='', lw=3, markersize=15, markerfacecolor='none',
                markeredgewidth=3, alpha=0.6, label='Model photometry')
    ax.errorbar(weff, galphot, yerr=galphoterr, marker='o', ls='', lw=2, markersize=10,
                markeredgewidth=2, alpha=0.8, label='Observed photometry',
                elinewidth=2, capsize=5)
                
    ax.set_xlabel(r'Observed-Frame Wavelength (${}$m)'.format('\mu'))
    ax.set_ylabel('Flux (AB mag)')
    #ax.set_ylabel('Flux Density (mJy)')
    ax.set_xlim(minwave, maxwave)
    ax.set_ylim(minflux, maxflux)
    ax.set_xscale('log')
    ax.invert_yaxis()
    #ax.set_yscale('log')
    #ax.legend(loc='upper right', fontsize=16, frameon=True)
    # https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    #ax.get_xaxis().set_major_formatter(ScalarFormatter())
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

    # Add an inset with the posterior probability distribution.
    ax1 = fig.add_axes([0.23, 0.68, 0.22, 0.22])
    ax1.hist(chain[:, 4], bins=50, histtype='step', linewidth=2, 
             edgecolor='k',fill=True)    
    ax1.set_xlim(10.5, 11.5)
    ax1.set_yticklabels([])
    ax1.set_xlabel(r'$\log_{10}(\mathcal{M}/\mathcal{M}_{\odot})$')
    ax1.set_ylabel(r'$P(\mathcal{M})$')
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(16)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png)

def subtriangle(results, showpars=None, truths=None, start=0, thin=2,
                chains=slice(None), logify=None, extents=None, png=None,
                **kwargs):
    """Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset of
    the parameters.

    :param start:
        The iteration number to start with when drawing samples to plot.

    :param thin:
        The thinning of each chain to perform when drawing samples to plot.

    :param showpars:
        List of string names of parameters to include in the corner plot.

    :param truths:
        List of truth values for the chosen parameters
    
    """
    import corner as triangle

    # Get the ull out the parameter names and flatten the thinned chains.
    parnames = np.array(results['theta_labels'])
    print(parnames)

    # Restrict to a particular set of parameters.
    if showpars:
        ind_show = np.array([parnames.tolist().index(p) for p in showpars])
        parnames = parnames[ind_show]
    else:
        ind_show = slice(None)

    # Get the arrays we need (trace, wghts)
    trace = results['chain'][..., ind_show]
    if trace.ndim == 2:
        trace = trace[None, :]
    trace = trace[chains, start::thin, :]
    wghts = results.get('weights', None)
    if wghts is not None:
        wghts = wghts[start::thin]
    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])

    # logify some parameters
    xx = samples.copy()
    if truths is not None:
        xx_truth = np.array(truths).copy()
    else:
        xx_truth = None
    if logify:
        for p in logify:
            if p in parnames:
                idx = parnames.tolist().index(p)
                xx[:, idx] = np.log10(xx[:,idx])
                parnames[idx] = "log({})".format(parnames[idx])
                if truths is not None:
                    xx_truth[idx] = np.log10(xx_truth[idx])

    # Make nice labels.
    niceparnames = _niceparnames(parnames)
        
    # mess with corner defaults
    corner_kwargs = {"plot_datapoints": False, "plot_density": False,
                     "fill_contours": True, "show_titles": True}
    corner_kwargs.update(kwargs)
    
    fig = triangle.corner(xx, labels=niceparnames, truths=xx_truth,
                          quantiles=[0.25, 0.5, 0.75], weights=wghts,
                          color='k', **corner_kwargs)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png)

def logmass2mass(logmass=11.0, **extras):
    return 10**logmass

def load_obs(seed=1, nproc=1, nmin=10, verbose=False, sps=None):
    """Load the photometry for NGC5322.
    
    """
    import sedpy
    from prospect.utils.obsutils import fix_obs    

    # photometry in maggies
    phot = dict(
        FUV=(3.63e-4*1e3/maggies2mJy, 1/(9.76e-6*1e3/maggies2mJy)**2),
        NUV=(1.87e-3*1e3/maggies2mJy, 1/(1.02e-5*1e3/maggies2mJy)**2),
        g=(51032.92*1e-9, 0.060252897*1e18),
        r=(108675.04*1e-9, 0.019527867*1e18),
        z=(173005.16*1e-9, 0.011034269*1e18),
        W1=(126294.73*1e-9, 0.1742277*1e18),
        W2=(67562.625*1e-9, 0.03782868*1e18),
        W3=(34738.555*1e-9, 0.000105459614*1e18),
        W4=(32283.23*1e-9, 2.8181848e-06*1e18))

    galex = ['galex_FUV', 'galex_NUV']
    ls = ['sdss_{}0'.format(b) for b in ['g', 'r', 'z']]
    wise = ['wise_w{}'.format(n) for n in ['1','2', '3', '4']]
    filternames = galex + ls + wise

    obs = {}
    obs['redshift'] = 0.0060
    obs["filters"] = sedpy.observate.load_filters(filternames)

    obs["maggies"] = np.array([phot[filt][0] for filt in phot.keys()])
    obs["maggies_unc"] = np.array([1/np.sqrt(phot[filt][1]) for filt in phot.keys()])

    # mask out W4
    #obs["phot_mask"] = np.array(['w4' in f.name for f in obs["filters"]])    
    
    # Create a handy vector of effective wavelengths (optional) 
    obs["phot_wave"] = [f.wave_effective for f in obs["filters"]]
    obs["wavelength"] = None # spectral wavelength
    obs["spectrum"] = None
    obs['unc'] = None  # spectral uncertainties are given here
    obs['mask'] = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    obs = fix_obs(obs)

    run_params = {}
    run_params['redshift'] = obs['redshift']
    run_params['verbose'] = verbose
    run_params['debug'] = False
    run_params['seed'] = seed
    run_params['nproc'] = nproc
    run_params['param_file'] = '' # no parameter file

    run_params['min_method'] = 'lm'
    run_params['nmin'] = 1
    
    if sps:
        run_params['sps_libraries'] = sps.ssp.libraries

    # dynesty Fitter parameters
    dyn_params = {
        'nested_bound': 'multi',  # bounding method
        'nested_sample': 'unif', # 'unif', 'slice' # sampling method
        'nested_nlive_init': 100,
        'nested_nlive_batch': 100,
        'nested_bootstrap': 0,
        'nested_dlogz_init': 0.05,
        'nested_weight_kwargs': {"pfrac": 1.0},
        #'nested_stop_kwargs': {"post_thresh": 0.05}
        }
    run_params.update(dyn_params)
    
    return obs, run_params

def load_sps(zcontinuous=1, verbose=False):
    """zcontinuous - interpolate between metallicity values.

    """
    from prospect.sources import CSPSpecBasis

    if verbose:
        print('Loading SPS models...', end='')
    t0 = time.time()
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    if verbose:
        print('...took {:.2f} sec'.format(time.time()-t0))
    return sps

def load_model(obs, template_library='delayed-tau', verbose=False):
    """
    http://dfm.io/python-fsps/current/stellarpop_api/#api-reference
    https://github.com/moustakas/siena-astrophysics/blob/master/research/redmapper/redmapper-stellar-mass.py#L125-L197    
    
    """
    from prospect.models import priors
    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models.transforms import dustratio_to_dust1

    def base_delayed_tau():
        model_params = TemplateLibrary['parametric_sfh']

        # Initialize with sensible numbers.
        model_params['tau']['init'] = 10.0
        model_params['tage']['init'] = 1.0
        model_params['logzsol']['init'] = 0.2

        # optimize log-stellar mass, not linear stellar mass
        model_params['logmass'] = {'N': 1, 'isfree': True, 'init': 11.0,
                                   'prior': priors.TopHat(mini=10.0, maxi=12.0),
                                   'units': '$M_{\odot}$'}

        model_params['mass']['isfree'] = False
        model_params['mass']['init'] = 10**model_params['logmass']['init']
        model_params['mass']['prior'] = None
        model_params['mass']['depends_on'] = logmass2mass
        
        # Adjust the prior ranges.
        model_params['tau']['prior'] = priors.LogUniform(mini=0.1, maxi=30.0)
        model_params['tage']['prior'] = priors.LogUniform(mini=0.01, maxi=10.0)
        model_params['logzsol']['prior'] = priors.TopHat(mini=-0.5, maxi=0.3)

        #print('HACK!!!!!!!!!!!!!')
        #model_params['tau']['isfree'] = False
        #model_params['tage']['isfree'] = False
        #model_params['logzsol']['isfree'] = False
        #model_params['dust2']['isfree'] = False

        return model_params

    if template_library == 'delayed-tau':
        # Underlying delayed tau model.
        model_params = base_delayed_tau()

    if template_library == 'bursty':
        # Underlying delayed tau model.
        model_params = base_delayed_tau()
        
        # Add bursts
        model_params.update(TemplateLibrary['burst_sfh'])
        
        model_params['fburst']['isfree'] = True
        model_params['fburst']['init'] = 0.1
        model_params['fburst']['prior'] = priors.TopHat(mini=0.0, maxi=1.0)

        model_params['fage_burst']['isfree'] = True
        model_params['fage_burst']['init'] = 0.9
        model_params['fage_burst']['prior'] = priors.TopHat(mini=0.5, maxi=1.0)

    # Add dust emission (with fixed dust SED parameters).
    model_params.update(TemplateLibrary['dust_emission'])

    model_params['dust2']['init'] = 1.0 # diffuse dust
    model_params['dust2']['prior'] = priors.TopHat(mini=0.0, maxi=4.0)

    # Add more dust flexibility.
    model_params['dust_type'] = {'N': 1, 'isfree': False, 'init': 0, 'units': 'dust model'}
    model_params['dust_index'] = {'N': 1, 'isfree': False, 'init': -0.7,
                                  'units': 'power-law index', 'prior': None}
    
    model_params['dust1'] = {'N': 1, 'isfree': False, 'init': 0.0, 'prior': None,
                             'units': 'optical depth towards young stars',
                             'depends_on': dustratio_to_dust1}
    model_params['dust_ratio'] = {'N': 1, 'isfree': True, 'init': 1.0,
                                  'prior': priors.TopHat(mini=1.0, maxi=10.0),
                                  'units': 'dust1/dust2 ratio (optical depth to young stars vs diffuse)'}

    ## Add nebular emission.
    #model_params.update(TemplateLibrary['nebular'])
    ##model_params['add_neb_continuum']['init'] = False
    #model_params['gas_logu']['init'] = -1.0 # harder radiation field [default is -2.0]

    # Fixed redshift.
    model_params['zred']['init'] = obs['redshift']
    model_params['zred']['isfree'] = False 

    # Change the IMF from Kroupa to Salpeter.
    model_params['imf_type']['init'] = 0
        
    # Now instantiate the model using this new dictionary of parameter specifications
    model = SedModel(model_params)
    if verbose:
        print(model)

    return model

def main():
    """
    Main wrapper script.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--priors', default='delayed-tau', type=str, choices=['delayed-tau', 'bursty'],
                        help='Choose the model priors.')
    parser.add_argument('--prefix', default='ngc5322', type=str, help='Output file prefix.')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generation.')
    parser.add_argument('--nproc', default=1, type=int, help='Number of cores to use.')
    parser.add_argument('--sedfit', action='store_true', help='Do the SED fit.')
    parser.add_argument('--qaplots', action='store_true', help='Make pretty plots.')
    parser.add_argument('--verbose', action='store_true', help='Be verbose.')
    args = parser.parse_args()

    ngc5322dir = os.path.join(os.getenv('LSLGA_DIR'), 'science', 'proposals', 'nasa-adap-2019')
    hfile = os.path.join(ngc5322dir, '{}-{}.h5'.format(args.prefix, args.priors))

    if args.sedfit:
        import prospect.io
        import prospect.fitting

        # Initialize the SPS library (takes a bit), the photometry, the "run
        # parameters" dictionary, and the model priors.
        sps = load_sps(verbose=args.verbose)
        obs, rp = load_obs(seed=args.seed, nproc=args.nproc, verbose=args.verbose, sps=sps)
        model = load_model(obs, args.priors, verbose=args.verbose)
        
        #with multiprocessing.Pool(args.nproc) as P:
        P = None
        output = prospect.fitting.fit_model(obs, model, sps, noise=(None, None),
                                            optimize=True, dynesty=False, emcee=False,
                                            #optimize=False, dynesty=True, emcee=False,
                                            #nested_posterior_thresh=0.05,
                                            pool=P, **rp)

        if os.path.isfile(hfile):
            os.remove(hfile)
        print('Writing {}'.format(hfile))
        prospect.io.write_results.write_hdf5(
            hfile, rp, model, obs, output['sampling'][0],
            output['optimization'][0], tsample=output['sampling'][1],
            toptimize=output['optimization'][1])

    if args.qaplots:
        from prospect.io import read_results as reader
        
        print('Reading {}...'.format(hfile), end='')
        t0 = time.time()
        result, obs, _ = reader.results_from(hfile, dangerous=False)
        print('...took {:.2f} sec'.format(time.time()-t0))

        # SED.
        sps = load_sps(verbose=args.verbose)
        model = load_model(obs, args.priors, verbose=args.verbose)

        png = os.path.join(ngc5322dir, '{}-{}-sed.png'.format(args.prefix, args.priors))
        #bestfit_sed(obs, chain=result['chain'], lnprobability=result['lnprobability'], 
        bestfit_sed(obs, chain=None, lnprobability=result['lnprobability'], 
                    sps=sps, model=model, seed=1, nrand=100, png=png)

        # Corner plot.
        png = os.path.join(ngc5322dir, '{}-{}-corner.png'.format(args.prefix, args.priors))
        subtriangle(result, showpars=['logmass', 'tage', 'tau', 'dust2', 'dust_ratio'],
                    logify=['tau'], png=png)

        #reader.subcorner(result, start=0, thin=1, fig=plt.subplots(5,5,figsize=(27,27))[0])

        pdb.set_trace()

if __name__ == '__main__':
    main()
