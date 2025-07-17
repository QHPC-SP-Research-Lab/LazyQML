from lazyqml.GUI._widgets import *

import ipywidgets

# Groups options in boxes
general_model_options_box = ipywidgets.VBox(
    [
        ipywidgets.Label("General model options"),
        epochs_widget,
        layers_widget,
        nqubits_widget,
        dataset_widget
    ], 
    layout=ipywidgets.Layout(width='50%', margin='10pt')
)

cv_options_box = ipywidgets.VBox(
    [
        ipywidgets.Label("Cross-validation options"),
        cv_checkbox,
        splits_widget,
        repeats_widget
    ],
    layout=ipywidgets.Layout(width='50%', margin='10pt')
)

classifiers_box = ipywidgets.VBox(
    [
        ipywidgets.Label("Classifiers"),
        qsvm_checkbox,
        qnn_checkbox,
        qnn_bag_checkbox
    ], 
    layout=ipywidgets.Layout(width='50%', margin='10pt')
)

ansatzs_box = ipywidgets.VBox(
    [
        ipywidgets.Label("Ansatzs"),
        hp_checkbox,
        tt_checkbox,
        two_checkbox,
        hwe_checkbox,
        annular_checkbox
    ],
    layout=ipywidgets.Layout(width='50%', margin='10pt')
)

embeddings_box = ipywidgets.VBox(
    [
        ipywidgets.Label("Embeddings"),
        rx_checkbox,
        ry_checkbox,
        rz_checkbox,
        zz_checkbox,
        amp_checkbox,
        dense_checkbox,
        ho_checkbox
    ],
    layout=ipywidgets.Layout(width='50%', margin='10pt')
)

bagging_box = ipywidgets.VBox(
    [
        ipywidgets.Label("Bagging options"),
        nestimators_widget,
        nsamples_widget,
        nfeatures_widget
    ],
    layout=ipywidgets.Layout(width='100%', margin='10pt')
)

other_options_box = ipywidgets.VBox(
    [
        ipywidgets.Label("Other options"),
        tn_widget,
        runs_widget,
        randomstate_widget,
        verbose_widget
    ],
    layout=ipywidgets.Layout(width='100%', margin='10pt')
)