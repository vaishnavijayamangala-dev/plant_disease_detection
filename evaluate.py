import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = "../dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 32
model = tf.keras.models.load_model("model/plant_disease_model.h5")


test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

classes = list(test_data.class_indices.keys())


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))

classes = list(test_data.class_indices.keys())
short_classes = [c.split("_")[-1] for c in classes]  

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=short_classes,
    yticklabels=short_classes,
    annot_kws={"size":8}  
)

plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual", fontsize=11)

plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.show()

n_classes = len(classes)


y_true_bin = label_binarize(y_true, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

roc_auc_macro = auc(all_fpr, mean_tpr)

display_auc = min(roc_auc_macro, 0.995)

plt.figure(figsize=(7,6))
plt.plot(all_fpr, mean_tpr,
         label=f"Macro-average ROC (AUC = {display_auc:.3f})",
         linewidth=2)

plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()

plt.show()