from seshat_classifier import seshat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def cm_custom(y_true, y_pred, display_labels=None, ax=None, cmap='Greys',cbar= True):
    # Get confusion matrix
    # display_labels = list(display_labels)
    # Make sure N/A is at the end of the list
    # if "N/A" in display_labels:
    #     display_labels.remove("N/A")
    #     display_labels.append("N/A")

    cm = confusion_matrix(y_true, y_pred, labels=display_labels)
    
    bool_zeros = (cm == 0)
    nan_columns = bool_zeros.all(axis=0)
    cm = cm[:, ~nan_columns]
    bool_zeros = (cm == 0)
    nan_rows = bool_zeros.all(axis=1)
    cm = cm[~nan_rows, :]

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Create custom annotations: normalized (first line), counts (second line)
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            norm_val = f"{cm_norm[i, j]:.2f}"
            count_val = f"{cm[i, j]}"
            annot[i, j] = f"{norm_val}\n{count_val}"


    # If no axis is given, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    
    # Protect the display_labels list from being over-written
    display_labels = display_labels.copy()
    
    # Edit labels based on length of classes
    if (len(display_labels) <= 3):
        if 'FS' in display_labels:
            display_labels[display_labels.index('FS')] = 'Field Star'
        if 'BD' in display_labels:  
            display_labels[display_labels.index('BD')] = 'Brown Dwarf'
        if 'WD' in display_labels:
            display_labels[display_labels.index('WD')] = 'White Dwarf'
        if 'Gal' in display_labels:
            display_labels[display_labels.index('Gal')] = 'Galaxy'
    if len(display_labels) > 3:
        if 'Contaminant' in display_labels:
            display_labels[display_labels.index('Contaminant')] = 'Cont.'
    
    display_labels = np.array(display_labels.copy())
    xticklabels = display_labels[~nan_columns]
    yticklabels = display_labels[~nan_rows]

    # Draw heatmap with custom annotations
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt='',
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        ax=ax,
        cbar_kws={'label': 'Normalized value'},
        cbar = cbar
    )
    

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    return ax

    
cosmos = pd.read_csv("~/Documents/Star_Formation/YSO+Classification/Synthetic_Data/Data/COSMOSWeb_Labeled.csv")
cosmos['Class'] = 'Gal'
cosmos.loc[cosmos.Label==3,'Class'] = 'BD'
# display_labels = list(np.unique(cosmos.Class.values))
display_labels = ['Gal','BD','WD','FS']
cosmos = seshat.classify(cosmos,cosmological=True,classes=display_labels,return_test=False,threads=6)

ax = cm_custom(cosmos.Class,cosmos.Predicted_Class,cmap='Greys',display_labels=display_labels)
plt.tight_layout()
plt.savefig("cosmos_cm_test.png")