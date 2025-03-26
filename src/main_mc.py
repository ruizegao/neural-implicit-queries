# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
import time
import argparse
import imageio
import jax
import numpy as np
import polyscope.imgui as psim
import jax.numpy as jnp

# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
import trimesh
import time
# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")

def do_hierarchical_mc(opts, implicit_func, params, n_mc_depth):
    data_bound = opts['data_bound']
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))

    print(f"do_hierarchical_mc {n_mc_depth}")

    tri_pos = hierarchical_marching_cubes(implicit_func, params, lower, upper, n_mc_depth, n_subcell_depth=3)
    tri_pos.block_until_ready()

    tri_inds = jnp.reshape(jnp.arange(3 * tri_pos.shape[0]), (-1, 3))
    tri_pos = jnp.reshape(tri_pos, (-1, 3))
    tri_pos = np.array(tri_pos)
    tri_inds = np.array(tri_inds)
    print(len(tri_pos))
    print(len(tri_inds))
    return trimesh.Trimesh(tri_pos, tri_inds)

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--mode", type=str, default='affine_fixed')
    parser.add_argument("--n_mc_depth", type=int, default=11)

    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    opts = queries.get_default_cast_opts()
    opts['data_bound'] = 1
    opts['res_scale'] = 1
    opts['tree_max_depth'] = 12
    opts['tree_split_aff'] = False
    opts['hit_eps'] = 1e-5
    mode = args.mode
    modes = ['sdf', 'interval', 'affine_fixed', 'affine_truncate', 'affine_append', 'affine_all', 'slope_interval',
             'crown', 'alpha_crown', 'forward+backward', 'forward', 'forward-optimized', 'dynamic_forward',
             'dynamic_forward+backward', 'affine+backward']
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 8
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    affine_opts['crown'] = 1.
    affine_opts['alpha_crown'] = 1.
    affine_opts['forward+backward'] = 1.
    affine_opts['forward'] = 1.
    affine_opts['forward-optimized'] = 1.
    affine_opts['dynamic_forward'] = 1.
    affine_opts['dynamic_forward+backward'] = 1.
    affine_opts['affine+backward'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = 'absolute'
    surf_color = (0.157, 0.613, 1.000)

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    # load the matcaps
    matcaps = render.load_matcap(os.path.join(ROOT_DIR, "assets", "matcaps", "wax_{}.png"))
    if mode == 'affine_truncate':
        # truncate options
        implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    elif mode == 'affine_append':
        # truncate options
        implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)
    elif mode == 'sdf':
            implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(args.input, mode=mode, **affine_opts)

    mc_mesh = do_hierarchical_mc(opts, implicit_func, params, n_mc_depth=args.n_mc_depth)
    mc_mesh.show()
    mc_mesh.export(args.output)


if __name__ == '__main__':
    main()
