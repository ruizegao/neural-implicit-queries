import sys

import jax
import jax.numpy as jnp

from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import utils
import render
import geometry
from bucketing import *
import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
from kd_tree import *
from jax.scipy import stats
import time


# =============================================================
# ==== Cast Rays
# =============================================================

def get_default_cast_opts():
    d = {
        'hit_eps': 0.001,
        'max_dist': 10.,
        'n_max_step': 512,
        'n_substeps': 1,
        'safety_factor': 0.98,
        'interval_grow_fac': 1.5,
        'interval_shrink_fac': 0.5,
        'interval_init_size': 0.1,  # relative, as afactor of max_dist
        'refine_width_fac': 2.,
        'n_side_init': 16,
    }
    return d


def primary_dim(vec):
    return jnp.argmax(jnp.abs(vec), keepdims=True)


def find_ray_plane_intersection_with_depth(plane, plane_dim, root, dir):
    t = (plane - root[plane_dim]) / dir[plane_dim]
    return root + t * dir, t


def find_rays_plane_intersection_with_depth(roots, dirs, plane, plane_dim):
    return jax.jit(jax.vmap(partial(find_ray_plane_intersection_with_depth, plane, plane_dim)))(roots, dirs)


def cast_rays_tree_based(
        func_tuple,
        params_tuple,
        roots,
        dirs,
        branching_method: str = "naive",
        delta=0.001,
        batch_size=None,
        enable_clipping=False
):
    """

    kD Tree based method for ray casting.

    :param func_tuple:
    :param params_tuple:
    :param roots:
    :param dirs:
    :param branching_method:    Specified branching heuristic to decide which dimension to split upon
    :param delta:
    :param batch_size:          If not None, nodes are processed in batches
    :param enable_clipping:     If true, builds a non-uniform kD tree using clipping
    :return:
    """
    split_depth = 3 * 6
    func = func_tuple[0]
    params = params_tuple[0]
    data_bound = 1.
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))
    center = (lower + upper) / 2.
    # print("branching method: ", branching_method)
    # if enable_clipping:
    #     node_lower_tree, node_upper_tree, node_type_tree, split_dim_tree, split_val_tree = construct_full_non_uniform_unknown_levelset_tree(
    #         func, params, lower.unsqueeze(0), upper.unsqueeze(0), branching_method=branching_method, split_depth=split_depth, batch_size=batch_size)
    # else:
    t0 = time.time()
    node_lower_tree, node_upper_tree, node_type_tree, split_dim_tree, split_val_tree = construct_full_uniform_unknown_levelset_tree(
        func, params, jnp.expand_dims(lower, axis=0), jnp.expand_dims(upper, axis=0), split_depth=split_depth,
        batch_size=batch_size)
    t1 = time.time()
    print("tree building time: ", t1 - t0)
    t_out = jnp.zeros((dirs.shape[0],))
    hit_id_out = jnp.zeros((dirs.shape[0],))
    is_hit = jnp.full((dirs.shape[0],), False)
    not_hit = jnp.full((dirs.shape[0],), True)
    hit_node = jnp.full((dirs.shape[0],), False)
    miss_node = jnp.full((dirs.shape[0],), True)
    node_lower_last_layer = node_lower_tree[2 ** split_depth - 1: 2 ** (split_depth + 1) - 1]
    node_upper_last_layer = node_upper_tree[2 ** split_depth - 1: 2 ** (split_depth + 1) - 1]
    node_valid = jnp.logical_not(jnp.isnan(node_lower_last_layer[:, 0]))
    primary_dimension = stats.mode(jax.jit(jax.vmap(primary_dim))(dirs).flatten()).mode
    node_lower_unknown_leaf = node_lower_last_layer[node_valid]
    node_upper_unknown_leaf = node_upper_last_layer[node_valid]
    # for n_l, n_u in zip(node_lower_unknown_leaf, node_upper_unknown_leaf):
    #     print(n_l, n_u, func.bound_box(params, n_l, n_u))

    if roots[0][primary_dimension] > center[primary_dimension]:
        all_interaction_planes = jnp.flip(jnp.unique(node_upper_unknown_leaf[:, primary_dimension].flatten()), axis=0)
    else:
        all_interaction_planes = jnp.unique(node_lower_unknown_leaf[:, primary_dimension].flatten())

    leaf_mask = jnp.isnan(node_lower_tree[:, 0])

    def traverse_tree(points):
        # print("*******")
        node_idx = jnp.zeros((points.shape[0],)).astype(jnp.int32)
        while True:
            terminate = jnp.logical_and(leaf_mask[node_idx * 2 + 1], leaf_mask[node_idx * 2 + 2])
            if terminate.all():
                return node_type_tree[node_idx]
            split_mask = jnp.all((points < node_upper_tree[node_idx * 2 + 1]), axis=1)
            left_child_mask = jnp.logical_and(jnp.logical_not(terminate), split_mask)
            right_child_mask = jnp.logical_and(jnp.logical_not(terminate), jnp.logical_not(split_mask))
            node_idx = jnp.where(left_child_mask, node_idx * 2 + 1, node_idx)
            node_idx = jnp.where(right_child_mask, node_idx * 2 + 2, node_idx)

    for plane, next_plane in zip(all_interaction_planes, all_interaction_planes[1:]):
        rays_plane_intersection, t = find_rays_plane_intersection_with_depth(roots, dirs, (plane + next_plane) / 2,
                                                                             primary_dimension)
        point_types = traverse_tree(rays_plane_intersection)
        new_hit_node = point_types.astype(jnp.int32) == SIGN_UNKNOWN
        t_out = jnp.where(jnp.logical_and(miss_node, new_hit_node), t, t_out)
        hit_node = jnp.logical_or(hit_node, new_hit_node)
        miss_node = jnp.logical_not(hit_node)
    t2 = time.time()
    print("tree traversal time", t2 - t1)
    for step in range(1000):
        bad_node = jnp.logical_and(hit_node, not_hit)
        curr_ray_end = roots + (t_out.reshape(-1, 1) + delta) * dirs
        curr_output = jnp.zeros((dirs.shape[0],))
        curr_output = curr_output.at[bad_node].set(
            jax.jit(jax.vmap(partial(func, params)))(curr_ray_end[bad_node]).flatten())
        t_out = t_out.at[bad_node].set(t_out[bad_node] + delta * jnp.sign(curr_output[bad_node]))
        new_hit = jnp.logical_and(curr_output < 0., not_hit)
        is_hit = jnp.logical_or(new_hit, is_hit)
        not_hit = jnp.logical_not(is_hit)
        hit_id_out = hit_id_out.at[new_hit].set(1.)
    t3 = time.time()
    print("ray marching time", t3 - t2)
    return t_out, hit_id_out, jnp.zeros((dirs.shape[0],)), 0


@partial(jax.jit, static_argnames=("funcs_tuple", "n_substeps"), donate_argnums=(5, 6, 8, 9, 10, 11, 12))
def cast_rays_iter(funcs_tuple, params_tuple, n_substeps, curr_roots, curr_dirs, curr_t, curr_int_size, curr_inds,
                   curr_valid, curr_count, out_t, out_hit_id, out_count, opts):
    INVALID_IND = out_t.shape[0] + 1

    # Evaluate the function
    def take_step(substep_ind, in_tup):
        root, dir, t, step_size, is_hit, hit_id, step_count = in_tup

        can_step = ~is_hit  # this ensure that if we hit on a previous substep, we don't keep stepping

        step_count += ~is_hit  # count another step for this ray (unless concvered)

        # loop over all function in the list
        func_id = 1
        for func, params in zip(funcs_tuple, params_tuple):
            pos_start = root + t * dir
            half_vec = 0.5 * step_size * dir
            pos_mid = pos_start + half_vec
            box_type = func.classify_general_box(params, pos_mid, half_vec[None, :])

            # test if the step is safe
            can_step = jnp.logical_and(
                can_step,
                jnp.logical_or(box_type == SIGN_POSITIVE, box_type == SIGN_NEGATIVE)
            )

            # For convergence testing, sample the function value at the start the interval and start + eps
            pos_start = root + t * dir
            pos_eps = root + (t + opts['hit_eps']) * dir
            val_start = func(params, pos_start)
            val_eps = func(params, pos_eps)

            # Check if we converged for this func
            this_is_hit = jnp.sign(val_start) != jnp.sign(val_eps)
            hit_id = jnp.where(this_is_hit, func_id, hit_id)
            is_hit = jnp.logical_or(is_hit, this_is_hit)

            func_id += 1

        # take a full step of step_size if it was safe, but even if not we still inch forward
        # (this matches our convergence/progress guarantee)
        this_step_size = jnp.where(can_step, step_size, opts['hit_eps'])

        # take the actual step (unless a previous substep hit, in which case we do nothing)
        t = jnp.where(is_hit, t, t + this_step_size * opts['safety_factor'])

        # update the step size
        step_size = jnp.where(can_step,
                              step_size * opts['interval_grow_fac'],
                              step_size * opts['interval_shrink_fac'])
        step_size = jnp.clip(step_size, a_min=opts['hit_eps'])

        return (root, dir, t, step_size, is_hit, hit_id, step_count)

    # substepping
    def take_several_steps(root, dir, t, step_size):
        # Perform some substeps
        is_hit = False
        hit_id = 0
        step_count = 0
        in_tup = (root, dir, t, step_size, is_hit, hit_id, step_count)

        out_tup = jax.lax.fori_loop(0, n_substeps, take_step, in_tup)

        _, _, t, step_size, is_hit, hit_id, step_count = out_tup

        return t, step_size, is_hit, hit_id, step_count

    # evaluate the substeps on a all rays
    curr_t, curr_int_size, is_hit, hit_id, num_inner_steps \
        = jax.jit(jax.vmap(take_several_steps))(curr_roots, curr_dirs, curr_t, curr_int_size)

    curr_count += curr_valid * num_inner_steps

    # Test convergence
    is_miss = curr_t > opts['max_dist']
    is_count_terminate = curr_count >= opts['n_max_step']
    terminated = jnp.logical_and(
        jnp.logical_or(jnp.logical_or(is_hit, is_miss), is_count_terminate),
        curr_valid)

    # Write out finished rays
    write_inds = jnp.where(terminated, curr_inds, INVALID_IND)
    out_t = out_t.at[write_inds].set(curr_t, mode='drop')
    out_hit_id = out_hit_id.at[write_inds].set(hit_id, mode='drop')
    out_count = out_count.at[write_inds].set(curr_count, mode='drop')

    # Finished rays are no longer valid
    curr_valid = jnp.logical_and(curr_valid, ~terminated)
    N_valid = curr_valid.sum()

    return curr_t, curr_int_size, curr_valid, curr_count, out_t, out_hit_id, out_count, N_valid


def cast_rays(funcs_tuple, params_tuple, roots, dirs, opts):
    N = roots.shape[0]
    N_evals = 0  # all of the evaluations, INCLUDING those performed on unused padded array elements
    n_substeps = opts['n_substeps']

    # Outputs go here
    out_t = jnp.zeros(N)
    out_hit_id = jnp.zeros(N, dtype=int)

    # Working data (we will shrink this as the algorithm proceeds and rays start terminating)
    curr_roots = roots
    curr_dirs = dirs
    curr_t = jnp.zeros(N)
    curr_int_size = jnp.ones(N) * opts['interval_init_size'] * opts['max_dist']
    # curr_int_size = None # TODO don't technically need this for SDFs
    curr_inds = jnp.arange(0, N, dtype=int)  # which original ray this working ray corresponds to
    curr_valid = jnp.ones(N, dtype=bool)  # a mask of rays which are actually valid, in-progress rays

    # Also track number of evaluations
    out_count = jnp.zeros(N, dtype=int)
    curr_count = jnp.zeros(N, dtype=int)

    iter = 0
    while (True):

        iter += 1
        curr_t, curr_int_size, curr_valid, curr_count, out_t, out_hit_id, out_count, N_valid \
            = cast_rays_iter(funcs_tuple, params_tuple, n_substeps, curr_roots, curr_dirs, curr_t, curr_int_size,
                             curr_inds, curr_valid, \
                             curr_count, out_t, out_hit_id, out_count, opts)
        N_evals += curr_t.shape[0] * n_substeps

        N_valid = int(N_valid)
        if N_valid == 0:
            break

        if fits_in_smaller_bucket(N_valid, curr_valid.shape[0]):
            new_bucket_size = get_next_bucket_size(N_valid)
            curr_valid, empty_start, curr_roots, curr_dirs, curr_t, curr_int_size, curr_inds, curr_count = \
                compactify_and_rebucket_arrays(curr_valid, new_bucket_size, curr_roots, curr_dirs, curr_t,
                                               curr_int_size, curr_inds, curr_count)

    return out_t, out_hit_id, out_count, N_evals


@partial(jax.jit, static_argnames=("funcs_tuple", "n_substeps"), donate_argnums=(5, 7, 8, 9, 10, 11, 12, 13, 14))
def cast_rays_frustum_iter(
        funcs_tuple, params_tuple, cam_params, iter, n_substeps,
        curr_valid,
        curr_frust_range,
        curr_frust_t,
        curr_frust_int_size,
        curr_frust_count,
        finished_frust_range,
        finished_frust_t,
        finished_frust_hit_id,
        finished_frust_count,
        finished_start_ind,
        opts):
    N = finished_frust_range.shape[0] + 1
    INVALID_IND = N + 1

    root_pos, look_dir, up_dir, left_dir, fov_x, fov_y, res_x, res_y = cam_params
    gen_cam_ray = partial(render.camera_ray, look_dir, up_dir, left_dir, fov_x, fov_y)

    # x/y should be integer coordinates on [0,res], they are converted to angles internally
    def take_step(ray_xu_yu, ray_xu_yl, ray_xl_yu, ray_xl_yl, mid_ray, expand_fac, is_single_pixel, substep_ind,
                  in_tup):
        t, step_size, is_hit, hit_id, step_demands_subd, step_count = in_tup
        t_upper = t + step_size
        t_upper_adj = t_upper * expand_fac

        # Construct the rectangular (but not-axis-aligned) box enclosing the frustum
        right_front = (ray_xu_yu - ray_xl_yu) * t_upper_adj / 2
        up_front = (ray_xu_yu - ray_xu_yl) * t_upper_adj / 2
        source_range = jnp.stack((right_front, up_front), axis=0)

        can_step = ~is_hit  # this ensure that if we hit on a previous substep, we don't keep stepping
        step_count += ~is_hit  # count another step for this ray (unless concvered)

        center_mid = root_pos + 0.5 * (t + t_upper_adj) * mid_ray
        center_vec = 0.5 * (t_upper_adj - t) * mid_ray
        box_vecs = jnp.concatenate((center_vec[None, :], source_range), axis=0)

        # loop over all function in the list
        func_id = 1
        for func, params in zip(funcs_tuple, params_tuple):
            # Perform the actual interval test
            box_type = func.classify_general_box(params, center_mid, box_vecs)

            # test if the step is safe
            can_step = jnp.logical_and(
                can_step,
                jnp.logical_or(box_type == SIGN_POSITIVE, box_type == SIGN_NEGATIVE)
            )

            # For convergence testing, sample the function value at the start the interval and start + eps
            # (this is only relevant/useful once the frustum is a single ray and we start testing hits)
            pos_start = root_pos + t * mid_ray
            pos_eps = root_pos + (t + opts['hit_eps']) * mid_ray
            val_start = func(params, pos_start)
            val_eps = func(params, pos_eps)

            # Check if we converged for this func
            # (this is only relevant/useful once the frustum is a single ray and we start testing hits)
            this_is_hit = jnp.sign(val_start) != jnp.sign(val_eps)
            hit_id = jnp.where(this_is_hit, func_id, hit_id)
            is_hit = jnp.logical_or(is_hit, this_is_hit)

            func_id += 1

        # take a full step of step_size if it was safe, but even if not we still inch forward
        # the is_single_pixel ensures that we only inch forward for single-pixel rays, we can't
        # be sure it's safe to do so for larger frusta.
        # (this matches our convergence/progress guarantee)
        this_step_size = jnp.where(can_step, step_size, opts['hit_eps'] * is_single_pixel)

        # take the actual step (unless a previous substep hit, in which case we do nothing)
        t = jnp.where(is_hit, t, t + this_step_size * opts['safety_factor'])

        # update the step size
        step_size = jnp.where(can_step,
                              step_size * opts['interval_grow_fac'],
                              step_size * opts['interval_shrink_fac'])

        step_demands_subd = utils.logical_or_all((step_demands_subd, step_size < opts['hit_eps'], is_hit))

        step_size = jnp.clip(step_size, a_min=opts['hit_eps'])

        return t, step_size, is_hit, hit_id, step_demands_subd, step_count

    # substepping
    def take_several_steps(frust_range, t, step_size):
        # Do all of the frustum geometry calculation here. It doesn't change
        # per-substep, so might as well compute it before we start substepping.

        x_lower = frust_range[0]
        x_upper = frust_range[2]
        y_lower = frust_range[1]
        y_upper = frust_range[3]
        is_single_pixel = jnp.logical_and(x_lower + 1 == x_upper, y_lower + 1 == y_upper)

        # compute bounds as coords on [-1,1]
        # TODO it would be awesome to handle whole-pixel frustums and get a guarantee
        # about not leaking/aliasing geometry. However, in some cases the bounds cannot make
        # progress even a single-pixel sized frustum, and get stuck. We would need to handle
        # sub-pixel frustums to guarantee progress, which we do not currently support. For
        # this reason we treat each pixel as a point sample, and build frustums around those
        # instead. The difference is the -1 on the upper coords here.
        xc_lower = 2. * (x_lower) / (res_x + 1.) - 1.
        xc_upper = 2. * (x_upper - 1) / (res_x + 1.) - 1.
        yc_lower = 2. * (y_lower) / (res_y + 1.) - 1.
        yc_upper = 2. * (y_upper - 1) / (res_y + 1.) - 1.

        # generate rays corresponding to the four corners of the frustum
        ray_xu_yu = gen_cam_ray(xc_upper, yc_upper)
        ray_xl_yu = gen_cam_ray(xc_lower, yc_upper)
        ray_xu_yl = gen_cam_ray(xc_upper, yc_lower)
        ray_xl_yl = gen_cam_ray(xc_lower, yc_lower)

        # a ray down the center of the frustum
        mid_ray = 0.5 * (ray_xu_yu + ray_xl_yl)
        mid_ray_len = geometry.norm(mid_ray)
        mid_ray = mid_ray / mid_ray_len

        # Expand the box by a factor of 1/(cos(theta/2) to account for the fact that the spherical frustum extends a little beyond the naive linearly interpolated box.
        expand_fac = 1. / mid_ray_len

        # Perform some substeps
        is_hit = False
        hit_id = 0
        step_count = 0
        step_demands_subd = False
        in_tup = (t, step_size, is_hit, hit_id, step_demands_subd, step_count)

        take_step_this = partial(take_step, ray_xu_yu, ray_xu_yl, ray_xl_yu, ray_xl_yl, mid_ray, expand_fac,
                                 is_single_pixel)
        out_tup = jax.lax.fori_loop(0, n_substeps, take_step_this, in_tup)

        t, step_size, is_hit, hit_id, step_demands_subd, step_count = out_tup
        return t, step_size, is_hit, hit_id, step_demands_subd, step_count

    # evaluate the substeps on a all rays
    curr_frust_t, curr_frust_int_size, is_hit, hit_id, step_demands_subd, num_inner_steps \
        = jax.jit(jax.vmap(take_several_steps))(curr_frust_range, curr_frust_t, curr_frust_int_size)

    # Measure frustum area in pixels, use it to track counts
    x_lower = curr_frust_range[:, 0]
    x_upper = curr_frust_range[:, 2]
    y_lower = curr_frust_range[:, 1]
    y_upper = curr_frust_range[:, 3]
    frust_area = (x_upper - x_lower) * (y_upper - y_lower)
    curr_frust_count += curr_valid * num_inner_steps * (1. / frust_area)

    # only size-1 frusta actually get to hit
    is_hit = jnp.logical_and(is_hit, frust_area == 1)

    is_miss = curr_frust_t > opts['max_dist']
    is_count_terminate = iter >= opts['n_max_step']
    terminated = jnp.logical_and(jnp.logical_or(jnp.logical_or(is_hit, is_miss), is_count_terminate), curr_valid)

    # Write out finished rays
    target_inds = utils.enumerate_mask(terminated, fill_value=INVALID_IND) + finished_start_ind
    finished_frust_range = finished_frust_range.at[target_inds, :].set(curr_frust_range, mode='drop')
    finished_frust_t = finished_frust_t.at[target_inds].set(curr_frust_t, mode='drop')
    finished_frust_hit_id = finished_frust_hit_id.at[target_inds].set(hit_id, mode='drop')
    finished_frust_count = finished_frust_count.at[target_inds].set(curr_frust_count, mode='drop')
    curr_valid = jnp.logical_and(curr_valid, ~terminated)
    finished_start_ind += jnp.sum(terminated)

    # Identify rays that need to be split
    # TODO some arithmetic repeated with the function raycast
    width_x = 2 * jnp.sin(jnp.deg2rad(fov_x) / 2 * (x_upper - x_lower) / res_x) * curr_frust_t
    width_y = 2 * jnp.sin(jnp.deg2rad(fov_y) / 2 * (y_upper - y_lower) / res_y) * curr_frust_t
    can_subd = jnp.logical_or(curr_frust_range[:, 2] > (curr_frust_range[:, 0] + 1),
                              curr_frust_range[:, 3] > (curr_frust_range[:, 1] + 1))
    needs_refine = utils.logical_or_all((width_x > opts['refine_width_fac'] * curr_frust_int_size,
                                         width_y > opts['refine_width_fac'] * curr_frust_int_size,
                                         step_demands_subd))  # last condition ensure rays which hit but still need subd always get it
    needs_refine = jnp.logical_and(needs_refine, can_subd)
    needs_refine = jnp.logical_and(needs_refine, curr_valid)

    N_needs_refine = jnp.sum(needs_refine)
    N_valid = jnp.sum(curr_valid)

    return curr_valid, curr_frust_t, curr_frust_int_size, curr_frust_count, needs_refine, \
        finished_frust_range, finished_frust_t, finished_frust_hit_id, finished_frust_count, finished_start_ind, N_valid, N_needs_refine


# For all frusta specified by sub_mask, split to be half the size along one axis (chosen automatically internally).
# Creates sum(sub_mask) new frusta entries, in addition to updating the existing subd entries, all with half the size.
# All entries specified by sub_mask MUST have index width >1 along one dimension.
# Precondition: there must be space in the arrays to hold the new elements. This routine at most
# doubles the size, therefore this requires frust.shape[0]-empty_start_ind > (2*sum(sub_mask))
@partial(jax.jit, donate_argnums=(2, 3, 4))
def subdivide_frusta(sub_mask, empty_start_ind, valid_mask, curr_frust_range, arrs):
    # curr_frust_t, curr_frust_int_size):
    N = sub_mask.shape[-1]
    INVALID_IND = N + 1

    # TODO should probably just assume this
    sub_mask = jnp.logical_and(sub_mask, valid_mask)

    # Pick which direction to subdivide in
    x_gap = curr_frust_range[:, 2] - curr_frust_range[:, 0]
    y_gap = curr_frust_range[:, 3] - curr_frust_range[:, 1]
    # assumption: one of these gaps will always be nonempty
    subd_x = x_gap >= y_gap

    # Generate the new frustums (call the two of them 'A' and 'B')
    # (for the sake of vectorization, we generate these at all frustra, but will only use them # at the ones which are actually being split)
    x_mid = (curr_frust_range[:, 0] + curr_frust_range[:, 2]) / 2
    y_mid = (curr_frust_range[:, 1] + curr_frust_range[:, 3]) / 2
    split_x_hi_A = jnp.where(subd_x, x_mid, curr_frust_range[:, 2])
    split_x_lo_B = jnp.where(subd_x, x_mid, curr_frust_range[:, 0])
    split_y_hi_A = jnp.where(~subd_x, y_mid, curr_frust_range[:, 3])
    split_y_lo_B = jnp.where(~subd_x, y_mid, curr_frust_range[:, 1])
    frust_range_A = curr_frust_range
    frust_range_A = frust_range_A.at[:, 2].set(split_x_hi_A)
    frust_range_A = frust_range_A.at[:, 3].set(split_y_hi_A)
    frust_range_B = curr_frust_range
    frust_range_B = frust_range_B.at[:, 0].set(split_x_lo_B)
    frust_range_B = frust_range_B.at[:, 1].set(split_y_lo_B)

    arrs_out = arrs  # initially this is just a copy (since B arrays inherit all the same data)

    # Overwrite the new A frustum on to the original entry
    overwrite_A = sub_mask
    curr_frust_range = jnp.where(overwrite_A[:, None], frust_range_A, curr_frust_range)
    # curr_frust_t = jnp.where(overwrite_A, frust_t_A, curr_frust_t)  # optimization: this is a no-op
    # curr_frust_int_size = jnp.where(overwrite_A, frust_int_size_A, curr_frust_int_size)  # optimization: this is a no-op

    # Compactify the new B entries, then roll them to their starting position in the new array
    compact_inds = jnp.nonzero(sub_mask, size=N, fill_value=INVALID_IND)[0]
    frust_range_B = frust_range_B.at[compact_inds, :].get(mode='drop')
    frust_range_B = jnp.roll(frust_range_B, empty_start_ind, axis=0)

    # Prep data arrays corresponding to all the B frusta
    arrs_B = []
    for a in arrs:
        a = a.at[compact_inds, ...].get(mode='drop')
        a = jnp.roll(a, empty_start_ind, axis=0)
        arrs_B.append(a)
    overwrite_B = jnp.roll(compact_inds < INVALID_IND, empty_start_ind)
    # print(f"overwrite_B:\n{overwrite_B}")

    # Overwrite
    curr_frust_range = jnp.where(overwrite_B[:, None], frust_range_B, curr_frust_range)
    for i in range(len(arrs_out)):
        arrs_out[i] = jnp.where(overwrite_B, arrs_B[i], arrs_out[i])
    valid_mask = jnp.logical_or(valid_mask, overwrite_B)

    return valid_mask, curr_frust_range, arrs_out


@jax.jit
def frustum_needs_subdiv_to_pixel(frust_valid, frust_range):
    is_single_pixel = jnp.logical_and((frust_range[:, 0] + 1) == frust_range[:, 2],
                                      (frust_range[:, 1] + 1) == frust_range[:, 3])
    needs_subd = jnp.logical_and(frust_valid, ~is_single_pixel)
    return needs_subd, jnp.any(needs_subd)


@partial(jax.jit, static_argnames=("res_x", "res_y"))
def write_frust_output(res_x, res_y, finished_frust_range, finished_frust_t, finished_frust_count,
                       finished_frust_hit_id):
    ## (3) Write the result (one pixel per frustum)
    out_t = jnp.zeros((res_x, res_y))
    out_hit_id = jnp.zeros((res_x, res_y), dtype=int)
    out_count = jnp.zeros((res_x, res_y), dtype=int)

    # not needed, all are valid
    # INVALID_IND = res_x+res_y+1
    x_coords = finished_frust_range[:, 0]
    y_coords = finished_frust_range[:, 1]

    at_args = {'mode': 'promise_in_bounds', 'unique_indices': True}
    out_t = out_t.at[x_coords, y_coords].set(finished_frust_t, **at_args)
    out_count = out_count.at[x_coords, y_coords].set(finished_frust_count, **at_args)
    out_hit_id = out_hit_id.at[x_coords, y_coords].set(finished_frust_hit_id, **at_args)

    return out_t, out_count, out_hit_id


def cast_rays_frustum(funcs_tuple, params_tuple, cam_params, in_opts):
    root_pos, look_dir, up_dir, left_dir, fov_x, fov_y, res_x, res_y = cam_params

    # make sure everything is on the device
    cam_params = tuple([jnp.array(x) for x in cam_params])
    opts = {}
    for k, v in in_opts.items():
        if k == 'n_substeps':
            n_substeps = v
        else:
            opts[k] = jnp.array(v)

    N_out = res_x * res_y

    ## Steps:
    ## (1) March the frustra forward
    ## (1a) Take a step
    ## (1b) Split any frustra that need it
    ## (2) Once all frusta have terminated, subdivide any that need it until they are a single pixel
    ## (3) Write the result (one pixel per frustum)

    # TODO think about "subpixel" accuracy in this. Technically, we can guarantee that tiny points
    # never slip between rays.

    do_viz = False

    ## Construct a initial frustums
    N_side_init = opts['n_side_init']
    N_init_frust = N_side_init ** 2
    N_evals = 0

    # This creates a grid of tiles N_side_init x N_side_init
    # (assumption: N_side_init <= res)
    x_ticks = jnp.linspace(start=0, stop=res_x, num=N_side_init + 1, dtype=int)
    y_ticks = jnp.linspace(start=0, stop=res_y, num=N_side_init + 1, dtype=int)
    x_start = jnp.tile(x_ticks[:-1], N_side_init)
    x_end = jnp.tile(x_ticks[1:], N_side_init)
    y_start = jnp.repeat(y_ticks[:-1], N_side_init)
    y_end = jnp.repeat(y_ticks[1:], N_side_init)
    curr_frust_range = jnp.stack((x_start, y_start, x_end, y_end), axis=-1)

    # All the other initial data
    curr_frust_t = jnp.zeros(N_init_frust)
    curr_frust_int_size = jnp.ones(N_init_frust) * opts['interval_init_size'] * opts['max_dist']
    curr_frust_count = jnp.zeros(N_init_frust)
    curr_valid = jnp.ones(N_init_frust, dtype=bool)
    empty_start_ind = N_init_frust

    # As the frusta terminate, we will write them to the 'finished' catergory here.
    # Note: the upper bound of `N` is tight here, and we should never need to expand.
    finished_frust_range = jnp.zeros((N_out, 4), dtype=int)
    finished_frust_t = jnp.zeros((N_out,))
    finished_frust_hit_id = jnp.zeros((N_out,), dtype=int)
    finished_frust_count = jnp.zeros((N_out,))
    finished_start_ind = 0

    if do_viz:
        prev_viz_val = {}

    ## (1) March the frustra forward
    iter = 0
    N_valid = N_init_frust
    while (True):

        # Take a step
        N_evals += curr_frust_t.shape[0]
        curr_valid, curr_frust_t, curr_frust_int_size, curr_frust_count, needs_refine, \
            finished_frust_range, finished_frust_t, finished_frust_hit_id, finished_frust_count, finished_start_ind, N_valid, N_needs_refine = \
            cast_rays_frustum_iter(funcs_tuple, params_tuple, cam_params, iter, n_substeps, \
                                   curr_valid, curr_frust_range, curr_frust_t, curr_frust_int_size, curr_frust_count, \
                                   finished_frust_range, finished_frust_t, finished_frust_hit_id, finished_frust_count,
                                   finished_start_ind, \
                                   opts)

        iter += n_substeps

        N_valid = int(N_valid)
        N_needs_refine = int(N_needs_refine)

        if (N_valid == 0): break

        space_needed = N_valid + N_needs_refine
        new_bucket_size = get_next_bucket_size(space_needed)
        curr_bucket_size = curr_valid.shape[0]
        fits_in_smaller_bucket = new_bucket_size < curr_bucket_size
        needs_room_to_subdivide = empty_start_ind + N_needs_refine > curr_bucket_size
        if needs_room_to_subdivide or fits_in_smaller_bucket:
            # print(f"** COMPATCT AND REBUCKET {curr_bucket_size} --> {new_bucket_size}")

            curr_valid, empty_start_ind, curr_frust_range, curr_frust_t, curr_frust_int_size, curr_frust_count, needs_refine = compactify_and_rebucket_arrays(
                curr_valid, new_bucket_size, curr_frust_range, curr_frust_t, curr_frust_int_size, curr_frust_count,
                needs_refine)
            empty_start_ind = int(empty_start_ind)

        # Do the spliting for any rays that need it
        curr_valid, curr_frust_range, [curr_frust_t, curr_frust_int_size, curr_frust_count] = \
            subdivide_frusta(needs_refine, empty_start_ind, curr_valid, curr_frust_range,
                             [curr_frust_t, curr_frust_int_size, curr_frust_count])

        empty_start_ind += N_needs_refine

    ## (2) Once all frusta have terminated, subdivide any that need it until they are a single pixel
    ## TODO: consider that we could write output using forI loops instead
    finished_valid = jnp.arange(finished_frust_t.shape[-1]) < finished_start_ind

    # NOTE: we can alternately compute the number needed manually. Each subdivision round splits an axis in half, so the number of rounds is the max of log_2(width_x) + log_2(width_y). (A quick test showed this didn't help performance)
    # i_sub = 0
    while (True):

        # Any frustum whose area is greater than 1
        needs_subd, any_needs_subd = frustum_needs_subdiv_to_pixel(finished_valid, finished_frust_range)

        if not any_needs_subd:
            break

        # Split frusta
        finished_valid, finished_frust_range, [finished_frust_t, finished_frust_hit_id, finished_frust_count] = \
            subdivide_frusta(needs_subd, finished_start_ind, finished_valid, finished_frust_range,
                             [finished_frust_t, finished_frust_hit_id, finished_frust_count])
        finished_start_ind += jnp.sum(needs_subd)

        # NOTE: this will always yield exactly N frusta total (one per pixel), so there is no need to resize the 'finished' arrays

    ## (3) Write the result (one pixel per frustum)
    out_t, out_count, out_hit_id = write_frust_output(res_x, res_y, finished_frust_range, finished_frust_t,
                                                      finished_frust_count, finished_frust_hit_id)

    return out_t, out_hit_id, out_count, N_evals



