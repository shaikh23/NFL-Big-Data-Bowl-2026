"""Gateway notebook for NFL Big Data Bowl 2026 (play_id by play_id)"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import kaggle_evaluation.core.templates
from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType


class NFLGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str] | None = None):
        super().__init__(data_paths, file_share_dir=None)
        self.data_paths = data_paths
        self.set_response_timeout_seconds(300)
        self.row_id_column_name = 'id'
        self._expected_prediction_columns = ['x', 'y']

    def unpack_data_paths(self):
        if self.data_paths:
            self.competition_data_dir = self.data_paths[0]
        else:
            self.competition_data_dir = Path(__file__).parent.parent
        self.competition_data_dir = Path(self.competition_data_dir)

    def generate_data_batches(self):
        test = pl.read_csv(self.competition_data_dir / 'test.csv')
        test_input = pl.read_csv(self.competition_data_dir / 'test_input.csv')

        if self.row_id_column_name not in test.columns:
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f"Row ID column '{self.row_id_column_name}' not found in test files."
            )

        test = test.with_columns(
            pl.concat_str([pl.col('game_id'), pl.lit('_'), pl.col('play_id')]).alias('unique_play_id'),
        )
        test_input = test_input.with_columns(
            pl.concat_str([pl.col('game_id'), pl.lit('_'), pl.col('play_id')]).alias('unique_play_id'),
        )
        batch_ids = test.select('unique_play_id').unique(maintain_order=True).get_column('unique_play_id').to_list()
        for batch_id in batch_ids:
            test_batch = test.filter(pl.col('unique_play_id') == batch_id)
            test_input_batch = test_input.filter(pl.col('unique_play_id') == batch_id)
            yield (test_batch.drop('unique_play_id'), test_input_batch.drop('unique_play_id')), test_batch.select(self.row_id_column_name)

    def competition_specific_validation(self, prediction, row_ids, data_batch) -> None:
        if not isinstance(prediction, (pd.DataFrame, pl.DataFrame)):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'Prediction must be a pandas or polars DataFrame.')

        # normalize to pandas for uniform checks
        if isinstance(prediction, pl.DataFrame):
            pred = prediction.to_pandas()
        else:
            pred = prediction

        expected = list(self._expected_prediction_columns)

        if len(pred) != len(row_ids):
            raise GatewayRuntimeError(f'Prediction has {len(pred)} rows but batch has {len(row_ids)} row_ids.')
        if self.row_id_column_name in [str(c) for c in pred.columns]:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, f"Prediction must not include '{self.row_id_column_name}'.")

        got = [str(c) for c in pred.columns]
        got_set = set(got)
        expected_set = set(expected)

        if got_set != expected_set:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, f'Prediction columns must match {expected}, but got {got}.')
        if len(got) != len(expected):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'Prediction has duplicate columns or wrong count.')

        pred = pred[expected]

        for col in expected:
            if not pd.api.types.is_numeric_dtype(pred[col]):
                raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, f"Column '{col}' must be numeric.")
            if pred[col].isna().any():
                raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, f"Column '{col}' contains NaNs.")
        if not np.isfinite(pred[expected].values).all():
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'Prediction contains non-finite values.')


if __name__ == '__main__':
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        gateway = NFLGateway()
        gateway.run()
    else:
        print('Skipping run for now')
