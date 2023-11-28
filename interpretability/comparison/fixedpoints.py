import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.decomposition import PCA

from interpretability.comparison.utils import FixedPoints


def sort_fps(all_fps):
    # Get the xstar locations of the FPs
    xstars = np.array([fp.xstar for fp in all_fps])
    # Perform PCA on the xstar locations
    xstars = np.squeeze(xstars)
    pca = PCA(n_components=1)
    pc1_xstar = pca.fit_transform(xstars) * -1
    sort_inds = np.argsort(pc1_xstar[:, 0])

    # Reorder all_fps by the sorted indices
    return all_fps[sort_inds]


def find_fixed_points(
    model: pl.LightningModule,
    state_trajs: np.array,
    mode: str,
    inputs: np.array,
    n_inits=1024,
    noise_scale=0.01,
    learning_rate=1e-2,
    tol_q=1e-12,
    tol_dq=1e-20,
    tol_unique=1e-3,
    max_iters=10000,
    random_seed=0,
    do_fp_sort=False,
    device="cpu",
    seed=0,
    use_subspace=False,
    compute_jacobians=False,
    use_percent=False,
):

    assert mode in {"tt", "dt", "node", "node_resid"}
    # set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    state_trajs = state_trajs.to(device)

    state_trajs_np = state_trajs.detach().numpy()
    state_pca = PCA(n_components=3)
    state_pca.fit(state_trajs_np.reshape(-1, state_trajs_np.shape[-1]))
    # Seed PyTorch
    torch.manual_seed(random_seed)
    # Prevent gradient computation for the neural ODE
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Choose random points along the observed trajectories
    if len(state_trajs.shape) > 2:
        n_samples, n_steps, state_dim = state_trajs.shape
        state_pts = state_trajs.reshape(-1, state_dim)
        idx = torch.randint(n_samples * n_steps, size=(n_inits,), device=device)

    else:
        n_samples_steps, state_dim = state_trajs.shape
        state_pts = state_trajs
        idx = torch.randint(n_samples_steps, size=(n_inits,), device=device)
    states = state_pts[idx]
    inputs = inputs[idx]

    mat1 = torch.from_numpy(state_pca.components_).to(device)
    # Add Gaussian noise to the sampled points
    states = states + noise_scale * torch.randn_like(states, device=device)
    # Require gradients for the states
    states = states.detach()
    initial_states = states.detach().numpy()
    states.requires_grad = True
    # Create the optimizer
    opt = torch.optim.Adam([states], lr=learning_rate)

    # Run the optimization
    iter_count = 1
    q_prev = torch.full((n_inits,), float("nan"), device=device)
    while True:
        if mode == "tt":
            # input_size = input_cond.shape[1]
            # inputs = input_cond * torch.ones(n_inits, input_size, device=device)
            # add a dim to states
            _, F = model.model(inputs, states)
            q = 0.5 * torch.sum((F.squeeze() - states.squeeze()) ** 2, dim=1)
        elif mode == "dt":
            # input_size = 3
            # inputs = input_cond * torch.ones(n_inits, 1, input_size, device=device)
            # add a dim to states
            # states = states.unsqueeze(1)
            _, F = model.decoder(inputs, states)
            # F.requires_grad = True
            q = 0.5 * torch.sum((F.squeeze() - states.squeeze()) ** 2, dim=1)
        elif mode == "node":
            input_size = model.decoder.cell.input_size
            inputs = torch.zeros(n_inits, input_size, device=device)
            input_hidden = torch.cat([states, inputs], dim=1)
            if use_subspace:
                vf_vec = model.decoder.cell.vf_net(input_hidden)
                vf_pca = torch.matmul(vf_vec, mat1.T)
                q = 0.5 * torch.sum((vf_pca**2), dim=1)
            else:
                q = 0.5 * torch.sum(model.decoder.cell.vf_net(input_hidden) ** 2, dim=1)
        dq = torch.abs(q - q_prev)
        q_scalar = torch.mean(q)
        # Backpropagate gradients and optimize
        q_scalar.backward()
        opt.step()
        opt.zero_grad()

        # Detach evaluation tensors
        q_np = q.cpu().detach().numpy()
        dq_np = dq.cpu().detach().numpy()
        # Report progress
        if iter_count % 500 == 0:
            mean_q, std_q = np.mean(q_np), np.std(q_np)
            mean_dq, std_dq = np.mean(dq_np), np.std(dq_np)
            print(f"\nIteration {iter_count}/{max_iters}")
            print(f"q = {mean_q:.2E} +/- {std_q:.2E}")
            print(f"dq = {mean_dq:.2E} +/- {std_dq:.2E}")

        # Check termination criteria
        converged = np.all(np.logical_or(dq_np < tol_dq * learning_rate, q_np < tol_q))
        if iter_count > 1 and converged:
            print("Optimization complete to desired tolerance.")
            break
        if iter_count + 1 > max_iters:
            print("Maximum iteration count reached. Terminating.")
            break
        q_prev = q
        iter_count += 1
    # Collect fixed points

    qstar = q.cpu().detach().numpy()
    all_fps = FixedPoints(
        xstar=states.cpu().detach().numpy().squeeze(),
        x_init=initial_states,
        qstar=qstar,
        dq=dq.cpu().detach().numpy(),
        n_iters=np.full_like(qstar, iter_count),
        tol_unique=tol_unique,
    )

    # unique_fps = all_fps
    if do_fp_sort:
        all_fps = sort_fps(all_fps)
    unique_fps = all_fps.get_unique()
    plot_qstar = True

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    # unique_fps = all_fps
    if plot_qstar:
        clipQ = np.clip(unique_fps.qstar, 1e-15, None)
        ax.hist(np.log10(clipQ), bins=10)
        # Set x axes to log
        plt.savefig("qstar_hist1.png")

    # Reject FPs outside the tolerance
    if use_percent:
        tol_q = np.quantile(unique_fps.qstar, 0.5)
    best_fps = unique_fps.qstar < tol_q
    # best_fps = np.ones_like(unique_fps.qstar, dtype=bool)
    best_fps = FixedPoints(
        xstar=unique_fps.xstar[best_fps],
        x_init=unique_fps.x_init[best_fps],
        qstar=unique_fps.qstar[best_fps],
        dq=unique_fps.dq[best_fps],
        n_iters=unique_fps.n_iters[best_fps],
        tol_unique=tol_unique,
    )
    print(f"Found {len(best_fps.xstar)} unique fixed points.")
    if compute_jacobians:
        # Compute the Jacobian for each fixed point
        def J_func(x):
            if mode == "tt":
                # input_size = input_cond.shape[1]
                # inputs = input_cond * torch.ones(1, input_size, device=device)
                inputs = 0  # FIXME: this is a hack
                _, F = model.model(inputs, x)
                F = F.squeeze()

            elif mode == "dt":
                # input_size = 3
                # inputs = input_cond * torch.ones(1, 1, input_size, device=device)
                inputs = 0  # FIXME: this is a hack
                _, F = model.decoder(inputs, x)
                F = F.squeeze()

            elif mode == "node":
                input_size = model.decoder.cell.input_size
                inputs = torch.zeros(1, input_size, device=device)
                input_hidden = torch.cat([x, inputs], dim=1)
                F = 0.1 * model.decoder.cell.vf_net(input_hidden) + x

            return F.squeeze()

        all_J = []
        x = torch.tensor(best_fps.xstar, device=device)
        for i in range(best_fps.n):
            single_x = x[i, :]
            single_x = single_x.unsqueeze(0)
            J = torch.autograd.functional.jacobian(J_func, single_x)
            J = J.squeeze()
            all_J.append(J)
        # Recombine and decompose Jacobians for the whole batch
        if all_J:
            dFdx = torch.stack(all_J).cpu().detach().numpy()
            best_fps.J_xstar = dFdx
            best_fps.decompose_jacobians()

            return best_fps
        else:
            return []
    else:
        return best_fps
