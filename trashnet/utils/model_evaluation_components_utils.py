import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wandb

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from trashnet.utils.main_utils import color_text

import wandb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(
      model,
      tf_dataset,
      class_names,
      confusion_plot_path=None,
      classification_report_path=None,
      save_plot=True,
      save_report=True,
      normalize=False,  
      figsize=(6,4)
    ):
    y_true = []
    y_pred = []

    for images, labels in tqdm(tf_dataset, desc="Evaluating model", unit=" batches"):
        predictions = model.predict(images, verbose=0)
        y_true.extend(tf.squeeze(labels).numpy())
        if predictions.shape[1] == 1:
            y_pred.extend((predictions > 0.5).astype(int))
        else:
            y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"
    title = 'Confusion Matrix'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if save_plot:
        plot_dir_path = os.path.dirname(confusion_plot_path)
        os.makedirs(plot_dir_path, exist_ok=True)
        plt.savefig(confusion_plot_path)
        print(f"Confusion matrix plot saved to {color_text(confusion_plot_path)}")

    # Generate classification report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(report_dict)
    report_text = classification_report(y_true, y_pred, target_names=class_names) 

    report_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-Score", "Support"])
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):  # skip 'accuracy' key, focus on class metrics
            report_table.add_data(class_name, metrics["precision"], metrics["recall"], metrics["f1-score"], metrics["support"])

    # Save report as plain text locally
    if save_report:
        classification_report_dir_path = os.path.dirname(classification_report_path)
        os.makedirs(classification_report_dir_path, exist_ok=True)
        with open(str(classification_report_path), 'w') as f:
            f.write(report_text)
        print(f"Classification report saved to {color_text(classification_report_path)}")

    # Log to WandB
    wandb.log({"Confusion Matrix": wandb.Image(str(confusion_plot_path))})
    wandb.log({"Classification Report": report_table})  # Log as dictionary for WandB


# def evaluate_model(
#       model,
#       tf_dataset,
#       class_names,
#       confusion_plot_path=None,
#       classification_report_path=None,
#       save_plot=True,
#       save_report=True,
#       normalize=False,  # Tambahin opsi untuk normalisasi confusion matrix
#       figsize=(6,4)
#     ):
#     """
#     Evaluasi model dan hasilkan confusion matrix serta classification report untuk klasifikasi biner atau multiclass.

#     Args:
#     - model: Model yang sudah dilatih.
#     - tf_dataset: Dataset untuk evaluasi.
#     - class_names: Daftar nama kelas yang ada.
#     - confusion_plot_path: Path untuk menyimpan confusion matrix plot.
#     - classification_report_path: Path untuk menyimpan classification report.
#     - save_plot: Simpan confusion matrix plot jika True.
#     - save_report: Simpan classification report jika True.
#     - normalize: Jika True, confusion matrix akan di-normalisasi (skala 0.0 - 1.0).
#     """
#     # Ambil label asli dan prediksi model
#     y_true = []
#     y_pred = []

#     # Lakukan evaluasi pada setiap batch
#     for images, labels in tqdm(tf_dataset, desc='Evaluation'):
#         # Buat prediksi probabilitas untuk setiap gambar
#         predictions = model.predict(images, verbose=0)

#         # Ambil label asli
#         y_true.extend(tf.squeeze(labels).numpy())

#         # Cek apakah klasifikasi biner atau multiclass
#         if predictions.shape[1] == 1:
#             # Binary classification: ubah prediksi berdasarkan threshold 0.5
#             y_pred.extend((predictions > 0.5).astype(int))
#         else:
#             # Multiclass classification: ambil kelas dengan probabilitas tertinggi
#             y_pred.extend(np.argmax(predictions, axis=1))

#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     # Hitung confusion matrix
#     cm = confusion_matrix(y_true, y_pred)

#     # Normalisasi confusion matrix jika diperlukan
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         fmt = ".2f"  # format untuk nilai desimal
#     else:
#         fmt = "d"  # format untuk nilai integer
#     title = 'Confusion Matrix (Jumlah)'

#     # Tampilkan dan simpan confusion matrix plot
#     plt.figure(figsize=figsize)
#     sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
#     plt.title(title)
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')

#     if save_plot:
#         plot_path = confusion_plot_path
#         plot_dir_path = os.path.dirname(plot_path)
#         os.makedirs(plot_dir_path, exist_ok=True)
#         plt.savefig(plot_path)
#         print(f"Confusion matrix plot saved to {plot_path}")

#     # Cetak dan simpan classification report
#     report = classification_report(y_true, y_pred, target_names=class_names)
#     print("Classification Report:")
#     print(report)

#     if save_report:
#         classification_report_dir_path = os.path.dirname(classification_report_path)
#         os.makedirs(classification_report_dir_path, exist_ok=True)
#         with open(classification_report_path, 'w') as f:
#             f.write(report)
#         print(f"Classification report saved to {classification_report_path}")
