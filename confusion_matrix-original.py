# # import pandas as pd
# # path = '/home/hangnt/hangnt/FixRes-master/save_folder/confusion_matrix.csv'
# # confusion_matrix = pd.read_csv(path, header = None)
# # columns = confusion_matrix.columns
# # for col in columns:
# #     confusion_matrix[col] = confusion_matrix[col].astype(int)
# # confusion_matrix.to_csv(path, index=False)
#
# import matplotlib.pyplot as plt
# import itertools
# import numpy as np
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 80.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# # import tensorflow as tf
# # from numpy import random
# #
# # writer_1 = tf.summary.FileWriter("./logs/plot_1")
# # writer_2 = tf.summary.FileWriter("./logs/plot_2")
# #
# # log_var = tf.Variable(0.0)
# # tf.summary.scalar("loss", log_var)
# #
# # write_op = tf.summary.merge_all()
# #
# # session = tf.InteractiveSession()
# # session.run(tf.global_variables_initializer())
# #
# # for i in range(100):
# #     # for writer 1
# #     summary = session.run(write_op, {log_var: random.rand()})
# #     writer_1.add_summary(summary, i)
# #     writer_1.flush()
# #
# #     # for writer 2
# #     summary = session.run(write_op, {log_var: random.rand()})
# #     writer_2.add_summary(summary, i)
# #     writer_2.flush()

import math

from matplotlib.colors import LogNorm
import numpy as np
np.random.seed(0)
import seaborn as sns
sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
#data = pd.read_csv('confusion_matrix.csv')
data = pd.read_csv('/home/hangnt/hangnt/FixRes-master/confusion_matrix.csv')
data = data.to_numpy()
print(data.min().min())
log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
print(log_norm)
cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(data.min().min()+1)), 1+math.ceil(math.log10(data.max().max())))]


g = sns.clustermap(data,
                   row_cluster=None, col_cluster=None,
                   vmin = data.min().min()+0.1, vmax=data.max().max(), cbar_kws={"ticks":cbar_ticks})


plt.savefig('demo.png')
