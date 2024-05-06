from pyrecon import IterativeFFTReconstruction, mpi, setup_logging
from pyrecon.utils import DistanceToRedshift, sky_to_cartesian, cartesian_to_sky
import mpytools as mpy
from cosmoprimo.fiducial import DESI
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import fitsio
from pathlib import Path
import matplotlib.pyplot as plt

def read_data():
    data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version}'
    data_fn = Path(data_dir) / f'{tracer}_{region}_clustering.dat.fits'
    return fitsio.read(data_fn)

def read_randoms(idx=0):
    data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version}'
    data_fn = Path(data_dir) / f'{tracer}_{region}_{idx}_clustering.ran.fits'
    return fitsio.read(data_fn)

def get_clustering_positions_weights(data):
    ra = data['RA']
    dec = data['DEC']
    dist = distance(data['Z'])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = data['WEIGHT']
    return pos, weights

def run_recon(recon_weights=False, fmesh=False):
    if fmesh:
        z = np.linspace(0.0, 5.0, 10000)
        # f = np.ones_like(z) * 0.824
        growth_at_dist = InterpolatedUnivariateSpline(distance(z), cosmo.growth_rate(z), k=3)
        f = 'mesh'
    else:
        growth_at_dist = None
        f = growth_rate[tracer]
    if recon_weights:
        nz_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version}/'
        nz_fn = Path(nz_dir) / f'{tracer}_{region}_nz.txt'
        data = np.genfromtxt(nz_fn)
        zmid = data[:, 0]
        nz = data[:, 3]
        n_at_dist = InterpolatedUnivariateSpline(distance(zmid), nz, k=3, ext=1)
    if mpicomm.rank == 0:
        data = read_data()
        data_positions, data_weights = get_clustering_positions_weights(data)
    else:
        data_positions, data_weights = None, None
    recon = IterativeFFTReconstruction(f=f, bias=bias, positions=data_positions,
                                    los='local', cellsize=cellsize, boxpad=1.1,
                                    position_type='pos', dtype='f8', mpicomm=mpicomm,
                                    mpiroot=0, growth_at_dist=growth_at_dist)
    recon.assign_data(data_positions, data_weights)
    for i in range(nrand):
        if mpicomm.rank == 0:
            randoms = read_randoms(i)
            random_positions, random_weights = get_clustering_positions_weights(randoms)
        else:
            random_positions, random_weights = None, None
        recon.assign_randoms(random_positions, random_weights)
    recon.set_density_contrast(smoothing_radius=smoothing_radius)
    if recon_weights:
        recon.set_optimal_weights(n_at_dist, P0)
    recon.run()
    data_positions_recon = recon.read_shifted_positions(data_positions)
    if mpicomm.rank == 0:
        dist, ra, dec = cartesian_to_sky(data_positions_recon)
        data['RA'], data['DEC'], data['Z'] = ra, dec, d2r(dist)
        if recon_weights and fmesh:
            output_dir = '/pscratch/sd/e/epaillas/recon_weights/recon_evo_weights/'
        elif recon_weights:
            output_dir = '/pscratch/sd/e/epaillas/recon_weights/recon_weights/'
        elif fmesh:
            output_dir = '/pscratch/sd/e/epaillas/recon_weights/recon_evo/'
        else:
            output_dir = '/pscratch/sd/e/epaillas/recon_weights/recon_vanilla/'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_fn = Path(output_dir) / f'{tracer}_{region}_recon.dat.fits'
        fitsio.write(output_fn, data, clobber=True)
    for i in range(nrand):
        if mpicomm.rank == 0:
            randoms = read_randoms(i)
            randoms_positions, randoms_weights = get_clustering_positions_weights(randoms)
        else:
            randoms_positions, randoms_weights = None, None
        randoms_positions_recon = recon.read_shifted_positions(randoms_positions)
        if mpicomm.rank == 0:
            dist, ra, dec = cartesian_to_sky(randoms_positions_recon)
            randoms['RA'], randoms['DEC'], randoms['Z'] = ra, dec, d2r(dist)
            output_fn = Path(output_dir) / f'{tracer}_{region}_{i}_recon.ran.fits'
            fitsio.write(output_fn, randoms, clobber=True)

setup_logging()
mpicomm = mpy.COMM_WORLD

bias = {'LRG': 2.0, 'QSO': 2.1}
P0 = {'LRG': 8.9e3, 'QSO': 5.0e3}
smoothing_radius = {'LRG': 15, 'QSO': 30}
growth_rate = {'LRG': 0.834, 'QSO': 0.928}

version = 'v1.2/blinded'
tracer = 'LRG'
region = 'NGC'
bias = bias[tracer]
P0 = P0[tracer]
smoothing_radius = smoothing_radius[tracer]
cellsize = 4.0
nrand = 18

cosmo = DESI()
distance = cosmo.comoving_radial_distance
d2r = DistanceToRedshift(distance)

run_recon(recon_weights=False, fmesh=False)
run_recon(recon_weights=False, fmesh=True)
run_recon(recon_weights=True, fmesh=True)

