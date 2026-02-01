import kaggle_evaluation.core.templates
import kaggle_evaluation.nfl_gateway


class NFLInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None):
        return kaggle_evaluation.nfl_gateway.NFLGateway(data_paths)
