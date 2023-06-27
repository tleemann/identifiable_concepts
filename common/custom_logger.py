import json
from datetime import datetime
import os, sys
class CustomLogger:
    """ A logger class for the experiments. """
    def __init__(self, root_path, run_params, update_interval = 20, exit_on_exist=False):
        """
            Parameters:
                root_path: path where the logs should be stored
                run_params: Dict with run parameters (will determine name of log file)
                update_interval: Determines how frequnetly the log file on the disk is updated.
                exit_on_exist: If true, the function terminates the programm if a complete log file ist found.

            To write all final data to the disk, please call the logger.close() function before terminating the program.
        """
        self.root_path = root_path

        self.myfile, data = self.get_log_file_path(run_params)

        if data is not None and "final_dirs" in data.keys():
            if exit_on_exist:
                print("Complete Record found. Exiting.")
                exit(0)
            else:
                print("Complete Record found. Partially Overwriting.")
                self.mydata = data
        else:
            print("Incomplete or no data found. Creating new run.")
            self.mydata = {}
            self.mydata["run_params"] = run_params
            self.mydata["start_time"] = str(datetime.now())
            self.mydata["values"] = []
        
        self.n_updates = 0
        self.update_interval = update_interval

    def update(self, metrics_dict):
        self.mydata["values"].append(metrics_dict)
        self.n_updates += 1
        if self.n_updates % self.update_interval == 0:
            self._dump()
        
    def close(self):
        """ Dump the remaining data and close the logger. """
        self._dump()

    def get_log_file_path(self, run_params):
        """ Return a path to a log file for this run 
            and create corresponding directories.
            return the path, and existing data, if a run has been previously found.
        """
        run_params_non_opt = {k: v for k,v in run_params.items()}
        if "optim" in run_params_non_opt: del run_params_non_opt["optim"]
        if "lr" in run_params_non_opt: del run_params_non_opt["lr"]
        if "batchsize" in run_params_non_opt: del run_params_non_opt["batchsize"]
        if "orth_samples" in run_params_non_opt: del run_params_non_opt["orth_samples"]
        #if "n_agg" in run_params_non_opt: del run_params_non_opt["n_agg"]

        run_str = [str(k)+ "-" + str(v) for (k,v) in run_params_non_opt.items()]
        run_str = "_".join(run_str)
        mypath = os.path.join(self.root_path, run_str)
        os.makedirs(mypath, exist_ok=True)
        rid = 0
        target_path = os.path.join(mypath, f"eval_results{rid}.json")
        if os.path.exists(target_path):
            data = json.load(open(os.path.join(mypath, f"eval_results{rid}.json")))
        else:
            data = None
        return os.path.join(mypath, f"eval_results{rid}.json"), data

    def _dump(self):
        with open(self.myfile, "w") as outfile:
            json.dump(self.mydata, outfile)

    def add_single_value(self, key, value):
        if type(value) == dict:
            if key not in self.mydata.keys():
                self.mydata[key] = value
            else:
                self.mydata[key].update(value)
        else:
            self.mydata[key] = value
        self._dump()