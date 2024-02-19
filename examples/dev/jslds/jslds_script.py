import jax.random as jr
from lfads_jslds.model import lfads_jslds_data, lfads_jslds_model, lfads_jslds_trainer

# specify dataset and create data object
system = "3BFF"
gen_model = "GRU_RNN"
n_neurons = 50
seed = 0
prefix = "20240216_NBFF_GRU_RNN_Final"
data = lfads_jslds_data(system, gen_model, n_neurons, seed, prefix)

data_dim = 50
ntimesteps = 500
ii_dim = 3
batch_size = 800
fp_reg = 10.0
out_nl_reg = 1.0
out_staylor_reg = 1.0
taylor_reg = 1.0
enc_dim = 128
con_dim = 128
gen_dim = 64
factors_dim = 10
mlp_nlayers = 2
mlp_n = 64
var_min = 0.001
l2reg = 0.00002
ic_prior_var = 0.1
prior = "ar"
ar_mean = 0.0
ar_autocorrelation_tau = 1e-10
ar_noise_variance = 0.1

num_batches = 25000
print_every = 200

step_size = 0.001
decay_factor = 0.9999
decay_steps = 1

keep_rate = 0.98

max_grad_norm = 5.0

kl_warmup_start = 500.0
kl_warmup_end = 1000.0
kl_min = 0.00
kl_max = 1.0
# initialize model object given data object
model = lfads_jslds_model(
    data_dim,
    ntimesteps,
    ii_dim,
    batch_size,
    fp_reg,
    out_nl_reg,
    out_staylor_reg,
    taylor_reg,
    enc_dim,
    con_dim,
    gen_dim,
    factors_dim,
    mlp_nlayers,
    mlp_n,
    var_min,
    l2reg,
    ic_prior_var,
    prior,
    ar_mean,
    ar_autocorrelation_tau,
    ar_noise_variance,
    num_batches,
    print_every,
    step_size,
    decay_factor,
    decay_steps,
    keep_rate,
    max_grad_norm,
    kl_warmup_start,
    kl_warmup_end,
    kl_min,
    kl_max,
)

# run model on batch of data
outputs = model.forward(
    data.train_data[:batch_size], data.train_inputs[:batch_size], jr.PRNGKey(0)
)

# train the model on the dataset
trainer = lfads_jslds_trainer()
trainer.set_model_and_data(model, data)
trainer.train()
