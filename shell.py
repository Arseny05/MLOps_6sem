from checker import BackgroundChecker
import cmd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import copy


class Shell(cmd.Cmd):
    prompt = ">>> "

    def __init__(self, bg_checker: BackgroundChecker):
        super().__init__()
        self.checker = bg_checker
        self.do_start_monitor("")
    
    def do_start_monitor(self, arg):
        self.checker.start()
        print("Background monitoring started\n")

    def help_start_monitor(self):
        print("start_monitor - starts background check for models drift\n")

    def do_stop_monitor(self, arg):
        self.checker.stop()
        print("Background monitoring stopped\n")

    def do_model_overview(self, arg):
        try:
            if arg.strip() == "all":
                models_list = list(self.checker.Orch.MODELS.keys())
                if not models_list:
                    print("No models have been initialized yet\n")
                    return
                print("=" * 60)
                for index in models_list:
                    self.checker.Orch.get_model_overview(index)
                print("=" * 60)
            else:
                print("=" * 60)
                self.checker.Orch.get_model_overview(int(arg))
                print("=" * 60)
        except Exception as e:
            print(f"Command failed due to {e}\n")
    
    def help_model_overview(self):
        print("model_overview <mode = int | 'all'> - get model or models overview\n")

    def do_dataset_overview(self, arg):
        try:
            if arg.strip() == "all":
                ds_list = list(self.checker.Orch.DATASETS.keys())
                if not ds_list:
                    print("No datasets have been loaded yet\n")
                    return
                print("=" * 60)
                for index in ds_list:
                    self.checker.Orch.get_dataset_overview(index)
                print("=" * 60)
            else:
                print("=" * 60)
                self.checker.Orch.get_dataset_overview(int(arg))
                print("=" * 60)
        except Exception as e:
            print(f"Command failed due to {e}\n")
    
    def help_dataset_overview(self):
        print("dataset_overview <mode = int | 'all'> - get dataset or datasets overview\n")

    def do_set_def_model(self, arg):
        try:
            if arg.strip() == "best":
                print(self.checker.Orch._set_def_model_("BEST"))
            else:
                print(self.checker.Orch._set_def_model_(int(arg.strip())))
        except Exception as e:
            print(f"Command failed due to {e}\n")

    def help_set_def_model(self):
        print("set_def_model <index = int | 'best'> - sets default model for training and inference\n")

    def do_set_dataset(self, arg):
        try:
            ds_id = int(arg.strip())
            print(self.checker.Orch._set_def_dataset_(ds_id))
        except Exception as e:
            print(f"Command failed due to {e}\n")

    def help_set_dataset(self):
        print("set_dataset <index = int> - sets current dataset for training and inference\n")

    def do_drift(self, arg):
        try:
            model_id = int(arg.strip())
            print(self.checker.Orch.check_for_drift(model_id))
        except Exception as e:
            print(f"Command failed due to {e}\n")
    
    def help_drift(self):
        print("drift <model_id> - manually starts check for model drift on all datasets for model with id = <model_id>\n")

    def do_update_model(self, arg):
        try:
            mode, logging, metrics, save_metrics = arg.strip().split()
            if mode.isdigit():
                index = int(mode)
            elif mode.lower() == 'best':
                index = "BEST"
            elif mode.lower() == "all":
                index = "all"
            else:
                raise ValueError(f"Unexpected parameter 'mode': {mode}\n")
            if logging.lower() == "true":
                logging = True
            elif logging.lower() == "false":
                logging = False
            else:
                raise ValueError(f"Unexpected parameter 'logging': {logging}\n")
            if metrics.lower() == "true":
                metrics = True
            elif metrics.lower() == "false":
                metrics = False
            else:
                raise ValueError(f"Unexpected parameter 'metrics': {metrics}\n")
            if save_metrics.lower() == "true":
                save_metrics = True
            elif save_metrics.lower() == "false":
                save_metrics = False
            else:
                raise ValueError(f"Unexpected parameter 'save_metrics': {save_metrics}\n")
            ds_id = self.checker.Orch.CUR_DS
            if ds_id not in self.checker.Orch.DATASETS:
                print("Current dataset has not been selected\n")
                return
            X, y = self.checker.Orch.DATASETS[ds_id]
            self.checker.Orch.update_model(index, X, y, logging, metrics, save_metrics)
        except Exception as e:
            print(f"Command failed due to {e}\n")
    
    def help_update_model(self):
        print("update_model <mode = 'all' | 'best' | int> <logging = bool> <metrics = bool> <save_mem_metrics = bool> - starts CV process on current dataset and creates new checkpoint for a model with id = <model_id>\n")

    def do_fit(self, arg):
        try:
            model_id, anew, save_metrics = arg.strip().split()
            model_id = int(model_id) if model_id != "best" else model_id.upper()
            if anew.lower() == "true":
                anew = True
            elif anew.lower() == "false":
                anew = False
            else:
                raise ValueError(f"Unexpected parameter 'anew': {anew}\n")
            if save_metrics.lower() == "true":
                save_metrics = True
            elif save_metrics.lower() == "false":
                save_metrics = False
            else:
                raise ValueError(f"Unexpected parameter 'save_metrics': {save_metrics}\n")
            ds_id = self.checker.Orch.CUR_DS
            if ds_id not in self.checker.Orch.DATASETS:
                print("Current dataset has not been selected\n")
                return
            X, y = self.checker.Orch.DATASETS[ds_id]
            filename = self.checker.Orch.fit_model(model_id, X, y, anew=anew, save_metrics=save_metrics)
            print(f"Checkpoint saved to {filename}\n")
        except Exception as e:
            print(f"Command failed due to {e}\n")
    
    def help_fit(self):
        print("fit <model_id = int | 'best'> <anew = bool> <save_mem_metrics = bool> - fits chosen model on a whole (!) current dataset\n")
    
    def do_eval(self, arg):
        try:
            model_id = arg.strip()
            index = model_id.upper() if model_id == 'best' else int(model_id)
            ds_id = self.checker.Orch.CUR_DS
            if ds_id not in self.checker.Orch.DATASETS:
                print("Current dataset has not been selected\n")
                return
            X, y = self.checker.Orch.DATASETS[ds_id]
            preds = self.checker.Orch.predict(index, X)
            mae, mse, r2 = mean_absolute_error(y, preds), mean_squared_error(y, preds), r2_score(y, preds)
            print(f"Metrics:\nMAE: {mae}\tMSE: {mse}\tR2: {r2}\n")
        except Exception as e:
            print(f"Command failed due to {e}\n")
    
    def help_eval(self):
        print("eval <model_id = int | 'best'> - evaluates chosen model on a whole (!) current dataset and prints main metrics\n")

    def do_add_dataset(self, arg):
        try:
            args = arg.strip().split()
            if len(args) == 1:
                print(self.checker.Orch.register_dataset(args[0]))
            elif len(args) == 2:
                if args[0].endswith(".db"):
                    print(self.checker.Orch.register_dataset(args[0], table=args[1]))
                else:
                    print(self.checker.Orch.register_dataset(args[0], index=int(args[1])))
            elif len(args) == 3:
                print(self.checker.Orch.register_dataset(args[0], args[1], int(args[2])))
            else:
                raise ValueError("Too many parameters!\n")
        except Exception as e:
            print(f"Command failed due to {e}\n")

    def help_add_dataset(self):
        print("add_dataset <data_path = str> <optional - table = str> <optional - index = int> - adds dataset from data_path. " \
        "If file extension is .db, parameter table is needed\n")

    def do_update_dataset(self, arg):
        try:
            args = arg.strip().split()
            if len(args) == 2:
                    print(self.checker.Orch.append_to_dataset(data_path=args[1], index=int(args[0])))
            elif len(args) == 3:
                print(self.checker.Orch.append_to_dataset(int(args[0]), args[1], args[2]))
            else:
                raise ValueError("Incorrect number of parameters!\n")
        except Exception as e:
            print(f"Command failed due to {e}\n")

    def help_update_dataset(self):
        print("update_dataset <index = int> <data_path = str> <optional - table = str> - updates dataset <index> from data_path. " \
        "If file extension is .db, parameter table is needed\n")

    def do_view_model_checkpoints(self, arg):
        try:
            index = int(arg.strip())
            self.checker.Orch.print_model_info(index)
        except Exception as e:
            print(f"Command failed due to {e}\n")
        
    def help_view_model_checkpoints(self):
        print("view_model_checkpoints <model_id = int> - prints all info about model with ID = model_id checkpoints with their indexes\n")

    def do_choose_model_checkpoint(self, args):
        try:
            index, version = list(map(int, args.strip().split()))
            print(self.checker.Orch.select_model_version(index, version))
        except Exception as e:
            print(f"Command failed due to {e}\n")

    def help_choose_model_checkpoint(self):
        print("choose_model_checkpoint <model_id = int> <version = int - sets model with ID = model_id to a checkpoint <version>\n")

    def do_explain_model(self, arg):
        try:
            model_id = arg.strip()
            index = model_id.upper() if model_id == "best" else int(model_id)
            print(self.checker.Orch.explain_model(index))
        except Exception as e:
            print(f"Command failed due to {e}\n")

    def help_explain_model(self):
        print("explain_model <model_id = int | 'best'> - explains model prediction on random sample\n")

    def do_auto_select(self, arg):
        try:
            ds_id = int(arg.strip())
            result = self.checker.Orch.auto_select_and_predict(ds_id)
            if result:
                print(f"Auto-selection completed: {result}\n")
        except Exception as e:
            print(f"Command failed due to {e}\n")

    def help_auto_select(self):
        print("auto_select <dataset_id> - automatically selects best model type and model, then evaluates it on dataset\n")

    def do_add_model(self, arg):
        try:
            model_type = arg.strip()
            default_params = copy.deepcopy(self.checker.Orch.config.get("models", {}).get(model_type, {}))
            if default_params == {}:
                raise ValueError("Unexpected model type!\n")
            for key in default_params:
                val = input(f"Input {key} value (default value is {default_params[key]}) or press enter to keep it default: ")
                if val.strip() != "":
                    default_params[key] = type(default_params[key])(val)
            v = input("Input desired model index or press enter to append it to a models list: ")
            index = None if v.strip() == "" else int(v)
            print(self.checker.Orch.model_creation(model_type, index, default_params))
        except Exception as e:
            print(f"Command failed due to {e}\n")

    def help_add_model(self):
        print("add_model <model_type = 'nn' | 'for' | 'tree' | 'sgd'> - creates model of model_type type\n")
    
    def do_delete_model(self, arg):
        try:
            index = int(arg.strip())
            print(self.checker.Orch.model_deletion(index))
        except Exception as e:
            print(f"Command failed due to {e}\n")
    
    def help_delete_model(self):
        print("delete_model <model_id = int> - deletes chosen model\n")
    
    def do_delete_dataset(self, arg):
        try:
            index = int(arg.strip())
            print(self.checker.Orch.dataset_deletion(index))
        except Exception as e:
            print(f"Command failed due to {e}\n")
    
    def help_delete_dataset(self):
        print("delete_dataset <dataset_id = int> - deletes chosen dataset\n")

    def do_exit(self, arg):
        self.checker.stop()
        return True

    def do_EOF(self, arg):
        print()
        self.checker.stop()
        return True