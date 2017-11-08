"""Plot the sheinberg results."""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


dd = '/media/data_cifs/contextual_circuit/sheinberg_results/sheinberg_data'
dirs = [
    'contextual_vector_conv3_3_sparse_pool_sheinberg_data_vgg16_3_2017_10_31_16_43_05.778448',
    'contextual_vector_separable_rand_init_conv3_3_sparse_pool_sheinberg_data_vgg16_3_2017_10_31_16_54_22.005238',
    'none_conv3_3_sparse_pool_sheinberg_data_vgg16_0_2017_10_31_17_04_40.316669'
]

all_data = []
for d in dirs:
    it_data = np.load(os.path.join(dd, d, 'data.npz'))['val_cv_out'].item()['scores']
    num_its = np.max(it_data.keys())
    all_data += [np.asarray([it_data[idx][0] for idx in range(num_its)])[:, None]]

all_data = np.asarray(all_data).squeeze().transpose(1, 0)  # .transpose().reshape(-1, 1)
labels = np.arange(all_data.shape[-1])[None, :].repeat(all_data.shape[0], axis=0).reshape(-1, 1)
time = np.arange(all_data.shape[0])[:, None].repeat(all_data.shape[-1], axis=-1).transpose().reshape(-1, 1)
all_data = all_data.transpose().reshape(-1, 1)
df = pd.DataFrame(
    np.hstack((all_data, labels, time)),
    columns=['data', 'label', 'time'])

f, ax = plt.subplots()
plot_data = sns.pointplot(
    data=df,
    x='time',
    y='data',
    hue='label',
    markers='.',
    scale=0.5,
    ax=ax,
    legend_out=True)
for item in plot_data.get_xticklabels():
    item.set_rotation(60)
n = 25
for index, label in enumerate(ax.xaxis.get_ticklabels()):
    if index % n != 0:
        label.set_visible(False)
label.set_visible(True)
new_title = 'VGG Conv3_3 with ImageNet weights.'
plot_data.legend_.set_title(new_title)
new_labels = dirs  # ['MLP', 'Contexual layer and MLP']
for t, l in zip(plot_data.legend_.texts, new_labels):
    t.set_text(l)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
plt.xlabel('Training epoch')
plt.ylabel('Pearson correlation fit for simulations\non held-out events')
plt.tight_layout()
plt.savefig(
    os.path.join(
        dd,
        'sheinberg_results.png'))
