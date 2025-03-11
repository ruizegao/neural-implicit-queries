# import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
import time
import argparse
import imageio
import jax
import polyscope.imgui as psim
import jax.numpy as jnp

# Imports from this project
import render, geometry, queries
from kd_tree import *
import implicit_mlp_utils
import time
import functools
# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def do_sample_surface(opts, implicit_func, params, n_samples, sample_width, n_node_thresh):
    data_bound = opts['data_bound']
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))

    rngkey = jax.random.PRNGKey(0)

    print(f"do_sample_surface n_node_thresh {n_node_thresh}")

    sample_points = sample_surface(implicit_func, params, lower, upper, n_samples, sample_width, rngkey, n_node_thresh=n_node_thresh)
    return  sample_points


def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)

    parser.add_argument("--mode", type=str, default='affine_all')
    parser.add_argument("--cast_frustum", action='store_true')
    parser.add_argument("--cast_tree_based", action='store_true')
    parser.add_argument("--res", type=int, default=1024)

    parser.add_argument("--image_write_path", type=str, default="render_out.png")

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
    cast_frustum = args.cast_frustum
    cast_tree_based = args.cast_tree_based
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

    n_samples = 1000
    n_node_thresh = 4096 * 64 * 8
    sample_width = 0.001
    sample_points = do_sample_surface(opts, implicit_func, params, n_samples=n_samples, sample_width=sample_width, n_node_thresh=n_node_thresh)
    sample_points = do_sample_surface(opts, implicit_func, params, n_samples=n_samples, sample_width=sample_width, n_node_thresh=n_node_thresh)
    sample_points = do_sample_surface(opts, implicit_func, params, n_samples=n_samples, sample_width=sample_width, n_node_thresh=n_node_thresh)
    print(sample_points.shape)
    func = functools.partial(implicit_func, params)
    print(jnp.abs(jax.vmap(func)(sample_points)).mean())

if __name__ == '__main__':
    main()
