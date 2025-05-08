import ipywidgets

cb_default_parameters = {
    "value": False,
    "disabled": False,
    "indent": False
}



### Widgets for user input with enhanced styles and multi-selection
epochs_widget = ipywidgets.IntText(
    min=1, 
    max=1000, 
    step=1, 
    value=10, 
    description="Epochs",
    indent=False,
    continuous_update=False
)

layers_widget = ipywidgets.IntText(
    value=7.5,
    min=0,
    max=100,
    step=1,
    description='Layers',
    disabled=False,
    indent=False,
    continuous_update=False
)

nqubits_widget = ipywidgets.IntText(
    min=1,
    max=30, 
    step=1, 
    value=4, 
    description="Qubits",
    indent=False,
    continuous_update=False
)

randomstate_widget = ipywidgets.IntText(
    value=1234, 
    description="Seed",
    indent=False,
    continuous_update=False
)

runs_widget = ipywidgets.IntText(
    min=1,
    max=10, 
    step=1, 
    value=1, 
    description="Runs",
    indent=False,
    continuous_update=False
)

dataset_widget = ipywidgets.Dropdown(
    options=['Iris', 'Breast Cancer', 'Wine'],
    value='Iris',
    description='Dataset',
    disabled=False,
)

# Cross validation options
cv_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Use k-fold cross-validation',
)

splits_widget = ipywidgets.IntText(
    min=1,
    max=50, 
    step=1, 
    value=4, 
    description="Folds",
    indent=False,
    continuous_update=False,
    disabled=not cv_checkbox.value
)

repeats_widget = ipywidgets.IntText(
    min=1,
    max=50, 
    step=1, 
    value=4, 
    description="Repeats",
    indent=False,
    continuous_update=False,
    disabled=not cv_checkbox.value
)

# Models
qsvm_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='QSVM'
)

qnn_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='QNN'
)

qnn_bag_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='QNN Bagging'
)

# Ansatzs
hp_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='HCzRx',
)

tt_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Tree Tensor',
)

two_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Two Local',
)

hwe_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Hardware Efficient',
)

annular_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Annular',
)

# Embeddings
rx_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Rotation X',
)

ry_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Rotation Y',
)

rz_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Rotation Z',
)

zz_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='ZZ',
)

amp_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Amplitude',
)

dense_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Dense Angle',
)

ho_checkbox = ipywidgets.Checkbox(
    **cb_default_parameters,
    description='Higher Order',
)

# Verbose switch
verbose_widget = ipywidgets.Checkbox(
    **cb_default_parameters, 
    description="Verbose"
)

tn_widget = ipywidgets.Dropdown(
    options=['State vector', 'Tensor Network (MPS)'],
    value='State vector',
    description='Sim. type',
    disabled=False
)

# Number of samples as a percentage (from 0.0 to 1.0)
nsamples_widget = ipywidgets.FloatSlider(
    value=1.0,
    min=0.0, 
    max=1.0, 
    step=0.01, 
    description="Samples",
    layout=ipywidgets.Layout(width='99%'),
    disabled=not qnn_bag_checkbox.value,
    continuous_update=False,
    readout_format='.0%'
)

nfeatures_widget = ipywidgets.FloatSlider(
    value=1.0,
    min=0.0, 
    max=1.0, 
    step=0.01, 
    description="Features",
    layout=ipywidgets.Layout(width='99%'),
    indent=False,
    disabled=not qnn_bag_checkbox.value,
    continuous_update=False,
    readout_format='.0%'
)

# Number of estimators (for QNN Bagging)
nestimators_widget = ipywidgets.IntText(
    min=1, 
    max=50, 
    step=1, 
    value=10, 
    description="Estimators",
    layout=ipywidgets.Layout(width='90%'),
    disabled=not qnn_bag_checkbox.value,
    continuous_update=False
)

all_widgets = [
    epochs_widget,
    layers_widget,
    nqubits_widget,
    randomstate_widget,
    runs_widget,
    dataset_widget,
    cv_checkbox,
    splits_widget,
    repeats_widget,
    qsvm_checkbox,
    qnn_checkbox,
    qnn_bag_checkbox,
    hp_checkbox,
    tt_checkbox,
    two_checkbox,
    hwe_checkbox,
    annular_checkbox,
    rx_checkbox,
    ry_checkbox,
    rz_checkbox,
    zz_checkbox,
    amp_checkbox,
    dense_checkbox,
    ho_checkbox,
    verbose_widget,
    tn_widget,
    nsamples_widget,
    nfeatures_widget,
    nestimators_widget
]


# Generate code button
generate_code_button = ipywidgets.Button(
    description='Generate code',
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    icon='check', # (FontAwesome names without the `fa-` prefix)
    layout=ipywidgets.Layout(width='50%')
)

save2file_button = ipywidgets.Button(
    description='Save to .py file',
    disabled=False,
    button_style='info',
    icon='download',
    layout=ipywidgets.Layout(width='50%')
)

#############

out_code = ipywidgets.Textarea(
    disabled=False,
    layout=ipywidgets.Layout(width='100%', height='250px')
)

output_code_display = ipywidgets.Output(style={'word-wrap': 'break-word'})
output_code_display.layout.display = 'flex'