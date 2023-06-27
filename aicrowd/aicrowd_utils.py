import os
import gin
import logging
import torch
from tqdm import tqdm
from common.posthoc_orthogonalizer import LinearTransformedNet_Stub
from common.posthoc_orthogonalizer import change_matrix_disentanglement, regression_matrix_disentanglement

def is_on_aicrowd_server():
    on_aicrowd_server = os.getenv('AICROWD_IS_GRADING', False)
    on_aicrowd_server = True if on_aicrowd_server != False else on_aicrowd_server
    return on_aicrowd_server


def get_gin_config(config_files, metric_name):
    for gin_eval_config in config_files:
        metric_name_of_config = gin_eval_config.split("/")[-1].replace(".gin", "")
        if metric_name == metric_name_of_config:
            return gin_eval_config
    return None


def evaluate_disentanglement_metric_with_linear_transformation(model, premult, 
        metric_names=['mig'], dataset_name='mpi3d_toy', ground_truth_dirs = None, factor_list=None):
    """ Evaluate a disentanglement metric on a linearly transformed model. 
        The output of the model (N) is left-multiplied by the D times N matrix premult.
        model: disentanglement model which returns an output of shape N for each input.
        premult: (D x N)-matrix which is left-multiplied to the output.
        metric_names: list of metrics.
        ground_truth_dirs: Ground truth directions of the concepts, required for the dir_score.
    """
    
    results_dict_all = {}
    metric_names_copy = [m for m in metric_names]
    A = premult.detach().cpu()

    # Treat discore seperately.
    if "dirscore" in metric_names_copy:
        dir_score =  change_matrix_disentanglement(A, ground_truth_dirs).item()
        logging.info(f"Direction Score:  {dir_score}")
        metric_names_copy.remove("dirscore")
        results_dict_all["eval_dirscore"] = dir_score
    if "dirscore_bin" in metric_names_copy:
        dir_score = regression_matrix_disentanglement(model, lambda x: x, A.t(), sample_epochs=200).item()
        logging.info(f"Direction Score (binary):  {dir_score}")
        results_dict_all["eval_dirscore_bin"] = dir_score
        metric_names_copy.remove("dirscore_bin")
    
    # Evaluate all other metrics on LinearTransformedNet.
    transformed_model = LinearTransformedNet_Stub(model, A)
    res_dlibmetrics = evaluate_disentanglement_metric(transformed_model, metric_names=metric_names_copy, factor_list=factor_list)
    print(res_dlibmetrics)
    results_dict_all.update(res_dlibmetrics)
    return results_dict_all


def evaluate_disentanglement_metric(model, metric_names=['mig'], dataset_name='mpi3d_toy', factor_list=None):
    # These imports are included only inside this function for code base to run on systems without
    # proper installation of tensorflow and libcublas
    from aicrowd import utils_pytorch
    from aicrowd.evaluate import evaluate
    from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1

    _study = unsupervised_study_v1.UnsupervisedStudyV1()
    evaluation_configs = sorted(_study.get_eval_config_files())
    evaluation_configs.append(os.path.join(os.getenv("PWD", ""), "extra_metrics_configs/irs.gin"))

    results_dict_all = dict()
    for metric_name in metric_names:
        if metric_name =="dirscore_bin" or metric_name == "dirscore":
            raise ValueError("dirscore_bin and dirscore are only supported during post-hoc disentanglement with a linear transformation.")
        elif metric_name == "accuracy":
            # Quick accuracy computation.
            model.pbar = tqdm(model.num_batches, position=0, leave=True)
            model.iter_test = 0
            correct = 0
            total = 0
            for _, x_true, label in model.data_loader:
                losses = dict()
                x_true = x_true.to(model.device)
                label = label.to(model.device)
                losses, fn_args = model.discriminative_forward(losses, x_true, label)
                ## Compute accuracy.
                total += len(fn_args["y_true"])
                correct += torch.sum(torch.argmax(fn_args["y_pred"], dim=1) == fn_args["y_true"].flatten()).item()
                #print(total, correct)
                model.iter_test += 1
                model.pbar.update(1)
                if(total > 20000):
                    break
            results = correct/total
            model.pbar = tqdm(model.num_batches, initial=model.iter, total = model.max_iter, position=0, leave=True)
        else:
            eval_bindings = [
                "evaluation.random_seed = {}".format(0),
                "evaluation.name = '{}'".format(metric_name)
            ]
            # Get the correct config file and load it
            my_config = get_gin_config(evaluation_configs, metric_name)
            if my_config is None:
                logging.warning('metric {} not among available configs: {}'.format(metric_name, evaluation_configs))
                return 0
            # gin.parse_config_file(my_config)
            gin.parse_config_files_and_bindings([my_config], eval_bindings)

            model_path = os.path.join(model.ckpt_dir, 'pytorch_model.pt')
            utils_pytorch.export_model(utils_pytorch.RepresentationExtractor(model.model.encoder, 'mean'),
                                    input_shape=(1, model.num_channels, model.image_size, model.image_size),
                                    path=model_path)

            output_dir = os.path.join(model.ckpt_dir, 'eval_results', metric_name)
            os.makedirs(os.path.join(model.ckpt_dir, 'results'), exist_ok=True)

            with gin.unlock_config():
                gin.bind_parameter("dataset.name", model.dset_name)
            results_dict = evaluate(model.ckpt_dir, output_dir, True, factor_list)
            gin.clear_config()
            results = 0
            for key, value in results_dict.items():
                if key != 'elapsed_time' and key != 'uuid' and key != 'num_active_dims':
                    results = value
        logging.info('Evaluation   {}={}'.format(metric_name, results))
        print('Evaluation   {}={}'.format(metric_name, results))
        results_dict_all['eval_{}'.format(metric_name)] = results
    # print(results_dict)
    return results_dict_all
