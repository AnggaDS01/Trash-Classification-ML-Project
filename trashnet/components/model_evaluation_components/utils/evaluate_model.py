import os
import wandb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(
      model: tf.keras.Model = None,
      tf_dataset: tf.data.Dataset = None,
      class_names: list = None,
      confusion_plot_path: str = None,
      classification_report_path: str = None,
      save_plot: bool = True,
      save_report: bool = True,
      normalize: bool = False,  
      figsize: tuple = (6,4)
    ):
    y_true = []
    y_pred = []

    for images, labels in tf_dataset:
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
    title = 'Confusion Matrix (Jumlah)'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if save_plot:
        plt.savefig(confusion_plot_path)
        print(f"Confusion matrix plot saved to {confusion_plot_path}")

    # Generate classification report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names)  # Default text format

    # Save report as plain text locally
    if save_report:
        with open(classification_report_path, 'w') as f:
            f.write(report_text)
        print(f"Classification report saved to {classification_report_path}")

    # Convert to WandB table
    report_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-Score", "Support"])
    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict):  # skip 'accuracy' key, focus on class metrics
            report_table.add_data(class_name, metrics["precision"], metrics["recall"], metrics["f1-score"], metrics["support"])

    # Log to WandB
    wandb.log({"Confusion Matrix": wandb.Image(confusion_plot_path)})
    wandb.log({"Classification Report": report_table})  # Log as dictionary for WandB