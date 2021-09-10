import numpy as np
import matplotlib
import matplotlib.pyplot as plt

batches = ["32", "64", "128"]
dimension_out = ["128", "256", "512"]

metrics  = np.array([[71.48, 71.45, 70.08],
                    [71.07, 68.65, 67.18],
                    [70.23, 69.25, 68.2]])


fig, ax = plt.subplots()
im = ax.imshow(metrics)

# We want to show all ticks...
ax.set_xticks(np.arange(len(batches)))
ax.set_yticks(np.arange(len(dimension_out)))
# ... and label them with the respective list entries
ax.set_xticklabels(batches)
ax.set_yticklabels(dimension_out)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(dimension_out)):
    for j in range(len(batches)):
        text = ax.text(j, i, metrics[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()
