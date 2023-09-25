import numpy as np
import torch

from .utils import AdaptiveGradNormClip, AdaptiveLearningRate, FixedPoints


class FixedPointFinder:
    def __init__(
        self,
        rnn_cell,
        tol_q=1e-12,
        tol_dq=1e-20,
        max_iters=5000,
        #  do_rerun_q_outliers=False,
        #  outlier_q_scale=10.0,
        do_exclude_distance_outliers=True,
        outlier_distance_scale=10.0,
        tol_unique=1e-3,
        max_n_unique=np.inf,
        do_compute_jacobians=True,
        do_decompose_jacobians=True,
        dtype="float32",
        random_seed=0,
        verbose=True,
        #  super_verbose=False,
        n_iters_per_print_update=100,
        alr_hps={},
        agnc_hps={},
        adam_hps={"eps": 0.01},
        device="cuda",
    ):

        # Prevent gradient computation for RNN
        for parameter in rnn_cell.parameters():
            parameter.requires_grad = False
        # Send RNN to the appropriate device
        self.rnn_cell = rnn_cell.to(device=device)
        self.device = device
        # self.is_lstm = isinstance(rnn, torch.nn.LSTM)
        self.tol_q = tol_q
        self.tol_dq = tol_dq
        self.max_iters = max_iters
        self.do_exclude_distance_outliers = do_exclude_distance_outliers
        self.outlier_distance_scale = outlier_distance_scale
        self.tol_unique = tol_unique
        self.max_n_unique = max_n_unique
        self.do_compute_jacobians = do_compute_jacobians
        self.do_decompose_jacobians = do_decompose_jacobians
        self.torch_dtype = getattr(torch, dtype)
        self.np_dtype = getattr(np, dtype)
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.verbose = verbose
        self.n_iters_per_print_update = n_iters_per_print_update

        self.alr_hps = alr_hps
        self.agnc_hps = agnc_hps
        self.adam_hps = adam_hps

    def sample_states(self, state_traj, n_inits, noise_scale=0.0):

        n_samples, n_steps, state_dim = state_traj.shape
        # Choose random points along the observed trajectories
        state_pts = state_traj.reshape(-1, state_dim)
        idx = self.rng.randint(n_samples * n_steps, size=n_inits)
        states = state_pts[idx]
        # Add Gaussian noise to the sampled points
        states = states + noise_scale * self.rng.randn(*states.shape)

        return states

    def find_fixed_points(self, initial_states, inputs, cond_ids=None):

        # Perform initial search for fixed points
        n_inits = len(initial_states)
        self._print_if_verbose(
            f"Searching for fixed points from {n_inits} initial states."
        )
        all_fps = self._run_joint_optimization(
            initial_states, inputs, cond_ids=cond_ids
        )

        # Filter out duplicates after first round of optimization
        unique_fps = all_fps.get_unique()
        self._print_if_verbose(f"Identified {unique_fps.n} unique fixed points.")

        # Optionally exclude fixed points that are too far away
        if self.do_exclude_distance_outliers:
            unique_fps = self._exclude_distance_outliers(unique_fps, initial_states)

        # TODO: Add additional iterations on outliers - first need
        # sequential optimization

        # # Optionally run additional optimization iterations on identified
        # # fixed points with q values on the large side of the q-distribution.
        # if self.do_rerun_q_outliers:
        #     unique_fps = \
        #         self._run_additional_iterations_on_outliers(unique_fps)
        #     unique_fps = unique_fps.get_unique()

        # Optionally subselect from the unique fixed points (e.g., for
        # computational savings when not all are needed.)
        if unique_fps.n > self.max_n_unique:
            self._print_if_verbose(
                "\tRandomly selecting %d unique "
                "fixed points to keep." % self.max_n_unique
            )
            max_n_unique = int(self.max_n_unique)
            idx_keep = self.rng.choice(unique_fps.n, max_n_unique, replace=False)
            unique_fps = unique_fps[idx_keep]

        if self.do_compute_jacobians:
            if unique_fps.n > 0:

                self._print_if_verbose(
                    "\tComputing input and recurrent "
                    "Jacobians at %d unique fixed points." % unique_fps.n
                )
                dFdx, dFdu = self._compute_jacobians(unique_fps)
                unique_fps.J_xstar = dFdx
                unique_fps.dFdu = dFdu

            else:
                # Allocate empty arrays, needed for robust concatenation
                n_states = unique_fps.n_states
                n_inputs = unique_fps.n_inputs

                shape_dFdx = (0, n_states, n_states)
                shape_dFdu = (0, n_states, n_inputs)

                unique_fps.J_xstar = unique_fps._alloc_nan(shape_dFdx)
                unique_fps.dFdu = unique_fps._alloc_nan(shape_dFdu)

            if self.do_decompose_jacobians:
                # self._test_decompose_jacobians(unique_fps, J_np, J_tf)
                unique_fps.decompose_jacobians(str_prefix="\t")

        return unique_fps, all_fps

    def _build_state_vars(self, initial_states, inputs):

        # Convert to tensors for Pytorch
        x = torch.tensor(
            initial_states,
            dtype=self.torch_dtype,
            device=self.device,
            requires_grad=True,
        )
        u = torch.tensor(inputs, dtype=self.torch_dtype, device=self.device)
        return x, u

    def _run_joint_optimization(self, initial_states, inputs, cond_ids=None):

        self._print_if_verbose("\tFinding fixed points " "via joint optimization.")
        # Create Pytorch Tensors
        x, u = self._build_state_vars(initial_states, inputs)
        # Create the optimizer
        opt = torch.optim.Adam([x], **self.adam_hps)
        # Create adaptive training objects
        alr = AdaptiveLearningRate(**self.alr_hps)
        agnc = AdaptiveGradNormClip(**self.agnc_hps)
        # Run the optimization
        iter_count = 1
        q_prev = torch.full((len(initial_states),), np.nan, device=self.device)
        while True:
            # Get adaptive training values
            iter_lr = alr()
            iter_gnc = agnc()
            # Perform forward-pass
            F = self.rnn_cell(u, x)
            q = 0.5 * torch.sum((F.squeeze() - x.squeeze()) ** 2, axis=1)
            dq = torch.abs(q - q_prev)
            q_scalar = torch.mean(q)
            # Backpropagate gradients
            q_scalar.backward()
            # Set learning rate and clip gradients
            for g in opt.param_groups:
                g["lr"] = iter_lr
            grad_norm = torch.nn.utils.clip_grad_norm_(x, iter_gnc)
            # Minimize q_scalar
            opt.step()
            opt.zero_grad()
            # Detach evaluation tensors
            q_np = q.cpu().detach().numpy()
            dq_np = dq.cpu().detach().numpy()
            # Report progress
            if np.mod(iter_count, self.n_iters_per_print_update) == 0:
                mean_q, std_q = np.mean(q_np), np.std(q_np)
                mean_dq, std_dq = np.mean(dq_np), np.std(dq_np)
                print(f"\nIteration {iter_count}/{self.max_iters}")
                print(f"q = {mean_q:.2E} +/- {std_q:.2E}")
                print(f"dq = {mean_dq:.2E} +/- {std_dq:.2E}")
            # Check termination criteria
            converged = np.all(
                np.logical_or(dq_np < self.tol_dq * iter_lr, q_np < self.tol_q)
            )
            if iter_count > 1 and converged:
                self._print_if_verbose("\tOptimization complete to desired tolerance.")
                break
            if iter_count + 1 > self.max_iters:
                self._print_if_verbose(
                    "\tMaximum iteration count reached. Terminating."
                )
                break

            q_prev = q
            alr.update(q_scalar.cpu().detach().numpy())
            agnc.update(grad_norm.cpu().detach().numpy())
            iter_count += 1

        # Detach outputs from PyTorch
        xstar = x.cpu().detach().numpy().squeeze()
        F_xstar = F.cpu().detach().numpy().squeeze()
        qstar = q.cpu().detach().numpy()
        dq = dq.cpu().detach().numpy()

        # Collect fixed points
        fps = FixedPoints(
            xstar=xstar,
            x_init=initial_states,
            inputs=inputs,
            # cond_id=cond_ids,
            F_xstar=F_xstar,
            qstar=qstar,
            dq=dq,
            n_iters=np.full_like(qstar, iter_count),
            tol_unique=self.tol_unique,
            dtype=self.np_dtype,
        )

        return fps

    def _compute_jacobians(self, fps):

        # Compute derivatives at the fixed points
        x, u = self._build_state_vars(fps.xstar, fps.inputs)

        def J_func(u, x):
            # Adjust dimensions and pass through RNN
            u = u[None, :]
            x = x[None, :]
            F = self.rnn_cell(u, x)
            return F.squeeze()

        # Compute the Jacobian for each fixed point
        all_J_rec = []
        all_J_inp = []
        for i in range(fps.n):
            single_x = x[i, :]
            single_u = u[i, :]
            # Simultaneously compute input and recurrent Jacobians
            J_inp, J_rec = torch.autograd.functional.jacobian(
                J_func, (single_u, single_x)
            )
            all_J_rec.append(J_rec)
            all_J_inp.append(J_inp)

        # Recombine Jacobians for the whole batch
        J_rec = torch.stack(all_J_rec).cpu().detach().numpy()
        J_inp = torch.stack(all_J_inp).cpu().detach().numpy()

        return J_rec, J_inp

    @staticmethod
    def identify_distance_non_outliers(fps, initial_states, dist_thresh):
        """Identify fixed points that are "far" from the initial states used
        to seed the fixed point optimization. Here, "far" means a normalized
        Euclidean distance from the centroid of the initial states that
        exceeds a specified threshold. Distances are normalized by the average
        distances between the initial states and their centroid.
        Empirically this works, but not perfectly. Future work: replace
        [distance to centroid of initial states] with [nearest neighbors
        distance to initial states or to other fixed points].
        Args:
            fps: A FixedPoints object containing optimized fixed points and
            associated metadata.
            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n_inits x n_states] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.
            dist_thresh: A scalar float indicating the threshold of fixed
            points' normalized distance from the centroid of the
            initial_states. Fixed points with normalized distances greater
            than this value are deemed putative outliers.
        Returns:
            A numpy array containing the indices into fps corresponding to the
            non-outlier fixed points.
        """

        # if tf_utils.is_lstm(initial_states):
        #     initial_states = \
        #         tf_utils.convert_from_LSTMStateTuple(initial_states)

        n_inits = initial_states.shape[0]
        n_fps = fps.n

        # Centroid of initial_states, shape (n_states,)
        centroid = np.mean(initial_states, axis=0)

        # Distance of each initial state from the centroid, shape (n,)
        init_dists = np.linalg.norm(initial_states - centroid, axis=1)
        avg_init_dist = np.mean(init_dists)

        # Normalized distances of initial states to the centroid, shape: (n,)
        scaled_init_dists = np.true_divide(init_dists, avg_init_dist)

        # Distance of each FP from the initial_states centroid
        fps_dists = np.linalg.norm(fps.xstar - centroid, axis=1)

        # Normalized
        scaled_fps_dists = np.true_divide(fps_dists, avg_init_dist)

        init_non_outlier_idx = np.where(scaled_init_dists < dist_thresh)[0]
        n_init_non_outliers = init_non_outlier_idx.size
        print(
            "\t\tinitial_states: %d outliers detected (of %d)."
            % (n_inits - n_init_non_outliers, n_inits)
        )

        fps_non_outlier_idx = np.where(scaled_fps_dists < dist_thresh)[0]
        n_fps_non_outliers = fps_non_outlier_idx.size
        print(
            "\t\tfixed points: %d outliers detected (of %d)."
            % (n_fps - n_fps_non_outliers, n_fps)
        )

        return fps_non_outlier_idx

    def _exclude_distance_outliers(self, fps, initial_states):
        """Removes putative distance outliers from a set of fixed points.
        See docstring for identify_distance_non_outliers(...).
        """

        idx_keep = self.identify_distance_non_outliers(
            fps, initial_states, self.outlier_distance_scale
        )
        return fps[idx_keep]

    def _print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
