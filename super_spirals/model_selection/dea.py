from typing import List, Tuple, Union
from functools import partial
import pandas as pd
import numpy as np
from scipy.optimize import linprog


def _unit_shadow_prices(model_metrics: pd.Series, peer_metrics: pd.DataFrame, greater_is_better: List[bool]) -> np.ndarray:
        greater_is_better_weight = np.where(greater_is_better, 1, -1)
        inputs_outputs = greater_is_better_weight* np.ones_like(
        peer_metrics
        )

        # outputs - inputs
        A_ub = inputs_outputs * peer_metrics
        b_ub = np.zeros(A_ub.shape[0])

        # \sum chosen model inputs = 1
        A_eq = np.where(greater_is_better_weight < 0.0, model_metrics, 0).reshape(1, -1)
        b_eq = np.array(1.0).reshape(1, -1)

        # max outputs == min -outputs
        c = np.where(greater_is_better_weight >= 0.0, -model_metrics, 0.0).reshape(1, -1)

        # compute dual
        dual_A_ub = np.vstack((A_ub, A_eq)).T
        dual_c = np.hstack((b_ub, b_eq.reshape(-1,))).T
        dual_b_ub = -c.T

        dual_result = linprog(
            dual_c,
            A_ub=dual_A_ub,
            b_ub=dual_b_ub,
            bounds=[(0, None) for _ in range(dual_A_ub.shape[1])]
        )

        shadow_ = dual_result.x[: peer_metrics.shape[0]]

        return shadow_.reshape(-1,)

def data_envelopment_analysis(
    validation_metrics: Union[pd.DataFrame, np.ndarray], greater_is_better: List = []
) -> pd.DataFrame:
    """
    :param validation_metrics: Metrics produced by __SearchCV
    :param greater_is_better: Whether that metric are to be considered inputs to decrease or outputs to increase
    :return: Shadow prices for comparing a model to is peers & Hypothetical Comparison Units to compare units
    """
    partialed_unit_shadow_scores = partial(_unit_shadow_prices, 
                                            peer_metrics = validation_metrics, 
                                            greater_is_better = greater_is_better)
    shadow_prices = pd.DataFrame(validation_metrics).apply(partialed_unit_shadow_scores, axis=1, result_type='expand')


    return shadow_prices
