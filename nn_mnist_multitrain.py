
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is intended to run multiple MNIST Models to
train and join the plots for all trained models
"""
import matplotlib.pyplot as plt
from nn_mnist_model import MNISTModel
import datetime

EPOCHS=200
BATCH_SIZE=128
models_layers_conf = [
    [],                                 # The default, with no hidden layers
    [[128,'relu']],                     # One hidden layer with 128 units and 'relu' activation
    [[128,'relu'],[64,'relu']],
    [[128,'relu'],[128,'relu']],
    [[128,'relu'],[128,'relu'],[64,'relu']],
]

models_list = []

for hidden_l in models_layers_conf:
    mnist_model = MNISTModel(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        auto_load_data=True
    )
    # Add the hidden layers
    mnist_model.build(hidden_layers=hidden_l)
    # compile the model
    mnist_model.compile()
    # Add the model into a dict an then to the list
    list_item = {
        "model": mnist_model,
        "name": mnist_model.descriptive_name
    }
    models_list.append(list_item)

# Train each model
for m_dict in models_list:
    m_dict["model"].fit()

# Plot All learning Curve
plot_rows = len(models_list)
plot_cols = 2
fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(16,8))
# Superior Title
fig.suptitle("Learning Curve", fontsize=18, fontweight='bold')
fig.tight_layout()
for i in range(plot_rows):
    offset_index = i * plot_cols
    ax_acc, ax_loss = axs.flat[offset_index:offset_index+2]
    models_list[i].add_learning_curve_plots(fig, ax_acc, ax_loss)
# Save the plots
plt.savefig('graficos_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))