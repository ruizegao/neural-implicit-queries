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
# Config

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")


def generate_camera_positions_on_grid(grid_size, radius, fov_deg):
    cameras = []
    theta_vals = jnp.linspace(0, jnp.pi, grid_size[0])
    phi_vals = jnp.linspace(0, 2 * jnp.pi, grid_size[1])

    for theta in theta_vals:
        for phi in phi_vals:
            # Calculate camera root position
            x = radius * jnp.sin(theta) * jnp.cos(phi)
            y = radius * jnp.sin(theta) * jnp.sin(phi)
            z = radius * jnp.cos(theta)
            root = jnp.array([x, y, z])

            # Look direction: pointing toward the origin
            look = -root / jnp.linalg.norm(root)

            # Arbitrary 'up' vector to avoid singularity at poles
            arbitrary_up = jnp.array([0.0, 0.0, 1.0]) if jnp.abs(look[2]) < 0.9 else jnp.array([0.0, 1.0, 0.0])

            # Left direction: perpendicular to look and arbitrary up
            left = jnp.cross(arbitrary_up, look)
            left /= jnp.linalg.norm(left)

            # Upward direction: perpendicular to look and left
            up = jnp.cross(look, left)
            up /= jnp.linalg.norm(up)

            cameras.append({
                'root': root,
                'look': look,
                'up': up,
                'left': left,
                'fov_deg': fov_deg
            })

    return cameras

def save_render_current_view(args, implicit_func, params, cast_frustum, cast_tree_based, opts, matcaps, surf_color):
    print(jax.devices())
    # root = jnp.array([2., 0., 0.])
    # look = jnp.array([-1., 0., 0.])
    # up = jnp.array([0., 1., 0.])
    # left = jnp.array([0., 0., 1.])
    root = jnp.array([0., -3., 0.])
    left = jnp.array([1., 0., 0.])
    look = jnp.array([0., 1., 0.])
    up = jnp.array([0., 0., 1.])
    fov_deg = 30
    # fov_deg = 60.
    res = args.res // opts['res_scale']

    surf_color = tuple(surf_color)
    # time_render_start = time.time()
    img, depth, count, _, eval_sum, raycast_time = render.render_image_naive(implicit_func, params, root, look, up,
                                                                             left, res, fov_deg, cast_frustum, opts,
                                                                             shading='matcap_color', matcaps=matcaps,
                                                                             shading_color_tuple=(surf_color,), tree_based=cast_tree_based)
    time_render_end = time.time()

    # flip Y
    img = img[::-1, :, :]

    # append an alpha channel
    alpha_channel = (jnp.min(img, axis=-1) < 1.) * 1.
    # alpha_channel = jnp.ones_like(img[:,:,0])
    img_alpha = jnp.concatenate((img, alpha_channel[:, :, None]), axis=-1)
    img_alpha = jnp.clip(img_alpha, a_min=0., a_max=1.)
    img_alpha = np.array(img_alpha * 255., dtype=np.uint8)
    time_post_process = time.time()
    print("Time post process:", time_post_process - time_render_end)
    print(f"Saving image to {args.image_write_path}")
    imageio.imwrite(args.image_write_path, img_alpha)
    print("image saving time:", time.time() - time_post_process)

def render_random_camera(args, implicit_func, params, cast_frustum, cast_tree_based, opts, matcaps, surf_color):
    res = args.res // opts['res_scale']

    cameras = generate_camera_positions_on_grid((1, 1), 2.5, 45.)
    root, look, up, left, fov_deg = cameras[0]['root'], cameras[0]['look'], cameras[0]['up'], cameras[0]['left'], cameras[0]['fov_deg']
    img, depth, count, _, eval_sum, raycast_time = render.render_image_naive(implicit_func, params, root, look, up,
                                                                             left, res, fov_deg, cast_frustum, opts,
                                                                             shading='matcap_color', matcaps=matcaps,
                                                                             shading_color_tuple=(surf_color,),
                                                                             tree_based=cast_tree_based)

    print("=====")
    imgs, rendering_times = [], []
    # cameras = generate_camera_positions(1000, (5., 6.), (30., 90.))
    cameras = generate_camera_positions_on_grid((5, 10), 2.5, 45.)
    for cam in cameras:
        root, look, up, left, fov_deg = cam['root'], cam['look'], cam['up'], cam['left'], cam['fov_deg']
        img, depth, count, _, eval_sum, raycast_time = render.render_image_naive(implicit_func, params, root, look, up,
                                                                                 left, res, fov_deg, cast_frustum, opts,
                                                                                 shading='matcap_color',
                                                                                 matcaps=matcaps,
                                                                                 shading_color_tuple=(surf_color,),
                                                                                 tree_based=cast_tree_based)

        imgs.append(np.array(img))
        rendering_times.append(raycast_time)

    avg_rendering_time = sum(rendering_times) / len(rendering_times)
    print("Average rendering time:", avg_rendering_time)
    # np.savez_compressed('birdcage_grid_cam_exact_1e-4.npz', imgs)
    return imgs

def main():
    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--mode", type=str, default='affine_fixed')
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
    opts['hit_eps'] = 1e-4
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

    # t0 = time.time()
    # save_render_current_view(args, implicit_func, params, cast_frustum, cast_tree_based, opts, matcaps, surf_color)
    # t1 = time.time()
    # print("time spent on ray casting: ", t1 - t0)
    # t0 = time.time()
    # save_render_current_view(args, implicit_func, params, cast_frustum, cast_tree_based, opts, matcaps, surf_color)
    # t1 = time.time()
    # print("time spent on ray casting: ", t1 - t0)
    # t0 = time.time()
    # save_render_current_view(args, implicit_func, params, cast_frustum, cast_tree_based, opts, matcaps, surf_color)
    # t1 = time.time()
    # print("time spent on ray casting: ", t1 - t0)
    results = render_random_camera(args, implicit_func, params, cast_frustum, cast_tree_based, opts, matcaps, surf_color)
    # np.savez_compressed('hammer_grid_cam_baseline_1e-4.npz', results)
    np.savez_compressed(args.output, results)


if __name__ == '__main__':
    main()
