#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import representation_analysis_tools.plots as plots
import representation_analysis_tools.logs as logs
import representation_analysis_tools.lazydict as lazydict
import representation_analysis_tools.rsa as rsa
import representation_analysis_tools.utils as repr_utils
import representation_analysis_tools.centered_kernel_alignment as cka
#%%

# Specify the two network to compare here
model_name = ['normal', 'local_0']

recompute = False

#%%
input_rdms = lazydict.RootPathLazyDictionary()
intrinsic_dims = lazydict.RootPathLazyDictionary()
outer_prod_triu_arrays = lazydict.RootPathLazyDictionary()

for model_name_ in model_name:
    input_rdms.update(logs.load_similarity_metric("input_rdms", model_name_))
    intrinsic_dims.update(logs.load_similarity_metric("intrinsic_dims", model_name_))
    outer_prod_triu_arrays.update(logs.load_similarity_metric("outer_prod_triu_arrays", model_name_))


#%%
model_name_ = '' if isinstance(model_name, list) or len(model_name) == 1 else model_name

input_rdms = repr_utils.separate_data_names(input_rdms)
corr_dist = rsa.corr_dist_of_input_rdms(input_rdms)
logs.log_distance_matrices(corr_dist, "corr_dist", model_name=model_name_)

#%%
model_name_ = '' if isinstance(model_name, list) or len(model_name) == 1 else model_name

outer_prod_triu_arrays_separated = repr_utils.separate_data_names(outer_prod_triu_arrays)
linear_cka_dist_mat = cka.matrix_of_linear_cka(outer_prod_triu_arrays_separated)
logs.log_distance_matrices(linear_cka_dist_mat, "linear_cka_dist_mat", model_name=model_name_)

#%%
mdm_embedding = repr_utils.repr_dist_embedding(corr_dist)

#%%
intrinsic_dims = repr_utils.separate_data_names(intrinsic_dims)
#%%
linear_cka_embedding = repr_utils.repr_dist_embedding(linear_cka_dist_mat)

# Plot the correlation dist matrix

fig, ax = plt.subplots(1,1)
img = ax.imshow(corr_dist['test'][1])
x_label_list = [x[2] for x in corr_dist['test'][0]]

ax.set_xticks([x for x in range(len(corr_dist['test'][0]))])
ax.set_yticks([x for x in range(len(corr_dist['test'][0]))])
ax.set_xticklabels(x_label_list)
ax.set_yticklabels(x_label_list)

plt.xticks(rotation=45)
fig.colorbar(img)
save_path = 'correlation_matrix_{}_{}.png'.format(model_name[0],model_name[1])
plt.savefig(save_path)
plt.show()

# Plot the cka matrix

fig, ax = plt.subplots(1,1)
img = ax.imshow(linear_cka_dist_mat['test'][1])
x_label_list = [x[2] for x in linear_cka_dist_mat['test'][0]]

ax.set_xticks([x for x in range(len(linear_cka_dist_mat['test'][0]))])
ax.set_yticks([x for x in range(len(linear_cka_dist_mat['test'][0]))])
ax.set_xticklabels(x_label_list)
ax.set_yticklabels(x_label_list)

plt.xticks(rotation=45)
fig.colorbar(img)
save_path = 'linear_cka_matrix_{}_{}.png'.format(model_name[0],model_name[1])
plt.savefig(save_path)
plt.show()


# Plot the Intrinsic Dimension

label = 'Intrinsic Dimension'
xs = []
ys = []
for name, value in intrinsic_dims['test'].items():
    xs.append(name[2])
    ys.append(value[0])

# Change this to the appropriate shape depending on the number of layers

ys1 = ys[:19]
ys2 = ys[19:]

fig, ax = plt.subplots()

# Change it here as well 

ax.plot(xs[:19], ys1[:19], label=model_name[0], c='r')
ax.plot(xs[:19], ys2[:19], label=model_name[1], c='b')

ax.set(xlabel='layer', ylabel='dimension')
ax.legend(loc='upper left')
ax.grid()
plt.xticks(rotation=45)
save_path = 'ID_normal_{}_{}.png'.format(model_name[0],model_name[1])
plt.savefig(save_path)
plt.show()


# Plot mdm embedding scatter plot

xs = []
ys = []
for value in mdm_embedding:
    xs.append(value[0])
    ys.append(value[1])

fig, ax = plt.subplots()
ax.scatter(xs,ys)
for i, txt in enumerate(corr_dist['test'][0][:19]):
    ax.annotate(txt[2], (xs[i], ys[i]))

ax.set(xlabel='mdm_emb_coord_1', ylabel='mdm_emb_coord_2')
ax.legend(loc='upper left')
ax.grid()
plt.xticks(rotation=45)
save_path = 'mdm_emb_coord_normal_local_1.png'
plt.savefig(save_path)
plt.show()

# Plot cka embedded scatter plot.

