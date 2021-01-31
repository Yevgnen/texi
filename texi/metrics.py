# -*- coding: utf-8 -*-

import collections


def multilabel_metrics(y, y_pred, kwargs=None):
    # pylint: disable=import-outside-toplevel
    from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

    if not kwargs:
        kwargs = collections.defaultdict(dict)

    metrics = {}
    metrics["hamming_loss"] = hamming_loss(y, y_pred)
    metrics["f1_score"] = f1_score(y, y_pred, **kwargs["f1_score"])
    metrics["precision_score"] = precision_score(y, y_pred, **kwargs["precision_score"])
    metrics["recall_score"] = recall_score(y, y_pred, **kwargs["recall"])

    return metrics


def clustering_report(X=None, y=None, y_pred=None, kwargs=None):
    # pylint: disable=import-outside-toplevel
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        calinski_harabasz_score,
        completeness_score,
        davies_bouldin_score,
        fowlkes_mallows_score,
        homogeneity_score,
        normalized_mutual_info_score,
        silhouette_score,
        v_measure_score,
    )
    from sklearn.metrics.cluster import contingency_matrix

    if not kwargs:
        kwargs = collections.defaultdict(dict)

    metrics = {}
    if y is not None and y_pred is not None:
        metrics["ajdusted_rank_score"] = adjusted_rand_score(y, y_pred)
        metrics["homogeneity_score"] = homogeneity_score(y, y_pred)
        metrics["completeness_score"] = completeness_score(y, y_pred)
        metrics["v_measure_score"] = v_measure_score(
            y, y_pred, **kwargs["v_measure_score"]
        )
        metrics["adjusted_mutual_info_score"] = adjusted_mutual_info_score(
            y, y_pred, **kwargs["adjusted_mutual_info_score"]
        )
        metrics["normalized_mutual_info_score"] = normalized_mutual_info_score(
            y, y_pred, **kwargs["normalized_mutual_info_score"]
        )
        metrics["fowlkes_mallows_score"] = fowlkes_mallows_score(
            y, y_pred, **kwargs["fowlkes_mallows_score"]
        )
        metrics["contingency_matrix"] = contingency_matrix(
            y, y_pred, **kwargs["contingency_matrix"]
        )

    if X is not None and y_pred is not None:
        metrics["silhouette_score"] = silhouette_score(
            X, y_pred, **kwargs["silhouette_score"]
        )
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, y_pred)
        metrics["davies_bouldin_score"] = davies_bouldin_score(X, y_pred)

    return metrics


def classification_report(*args, **kwargs):
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    from sklearn.metrics import classification_report as sklearn_classification_report

    kwargs["output_dict"] = True
    report = sklearn_classification_report(*args, **kwargs)
    report = pd.DataFrame(report).T

    return report


def confusion_matrix(*args, **kwargs):
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

    labels = kwargs.get("labels")
    confusion = sklearn_confusion_matrix(*args, **kwargs)
    confusion = pd.DataFrame(confusion, index=labels, columns=labels)

    return confusion


def confusion_matrix_fu(*args, **kwargs):
    def _plot_confusion_matrix(matrix):
        # pylint: disable=import-outside-toplevel
        import plotly.figure_factory as ff

        return ff.create_annotated_heatmap(
            z=matrix.values, x=labels, y=labels, showscale=True
        )

    kwargs.update({"normalize": None})
    confusion = confusion_matrix(*args, **kwargs).astype(int)

    kwargs.update({"normalize": "true"})
    normalized_confusion = confusion_matrix(*args, **kwargs).round(4)

    labels = kwargs.get("labels")
    if labels is not None:
        labels = list(labels)
    fig_confusion = _plot_confusion_matrix(confusion)
    fig_normalized_confusion = _plot_confusion_matrix(normalized_confusion)

    return confusion, normalized_confusion, fig_confusion, fig_normalized_confusion