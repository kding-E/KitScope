"""Layer 1 phishing-vs-benign detection API.

The implementation is re-exported from the historical ``second_layer`` module
for backward compatibility with earlier experiment artifacts.
"""

from .second_layer import (  # noqa: F401
    GBDTEnsembleBinaryClassifier,
    LogisticBinaryClassifier,
    MLPBinaryClassifier,
    SecondLayerConfig,
    SklearnBinaryClassifier,
    compute_binary_metrics,
    second_layer_from_dir,
    select_threshold_by_fpr,
    train_second_layer,
)
