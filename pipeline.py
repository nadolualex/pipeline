import os
import subprocess
import pandas as pd
import time
from datetime import datetime
from paths_config import *
from model_config import MODELS

EVAL_ONLY = False
DEMO = True

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def get_activation_command(env_name):
    return f"conda run -n {env_name}"

def has_output_images(output_dir, extensions):
    if not output_dir.exists(): 
        return False
    all_files = list(os.listdir(output_dir))
    for ext in extensions:
        for f in all_files:
            if f.lower().endswith(ext.lower()):
                return True
    return False

def run_model(model_name, config, dataset_key):
    current_dir = os.getcwd() # pwd
    try:
        # moving to the model directory
        model_dir = BASE_DIR / config["dir"]
        os.chdir(model_dir)
        # set the output directory
        output_dir = get_model_output_dir(model_name, dataset_key)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # set environment variables that are used within each model script
        os.environ['INPUT_DIR'] = str(get_dataset_input_dir(dataset_key))
        os.environ['GT_DIR'] = str(get_dataset_gt_dir(dataset_key))
        os.environ['OUTPUT_DIR'] = str(output_dir)
        os.environ['DATASET_KEY'] = dataset_key
        
        # create command 
        cmd = f"{get_activation_command(config['env'])} python {config['script']}"
        start_time = time.time()
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error running {model_name}:\n{stderr}")
            return None, "Error"
        
        runtime = time.time() - start_time
        return runtime, "Success"
    
    except Exception as e:
        print(f"Error running {model_name}: {str(e)}")
        return None, "Error"

    finally:
        # return to the root dir
        os.chdir(current_dir)

def run_evaluation(model_name, config, dataset_key):
    current_dir = os.getcwd()
    blank_metrics = {"PSNR": None, "F-measure": None, "PF-measure": None, "DRD": None, "NRM": None, "MPM": None}
    try:
        output_dir = get_model_output_dir(model_name, dataset_key)
        if not has_output_images(output_dir, config["extensions"]):
            print(f"No output images found for {model_name} in {output_dir} with extensions {config['extensions']}")
            return blank_metrics, None
        
        model_dir = BASE_DIR / config["dir"]
        os.chdir(model_dir)
        os.environ['INPUT_DIR'] = str(get_dataset_input_dir(dataset_key))
        os.environ['GT_DIR'] = str(get_dataset_gt_dir(dataset_key))
        os.environ['OUTPUT_DIR'] = str(output_dir)
        os.environ['DATASET_KEY'] = dataset_key
        cmd = f"{get_activation_command(config['env'])} python {config['eval_script']}"
        start_time = time.time()
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ)
        stdout, stderr = process.communicate()
        runtime = time.time() - start_time
        if process.returncode != 0:
            print(f"Error running evaluation for {model_name}:\n{stderr}")
            return blank_metrics, None
        
        metrics = blank_metrics.copy()
        found_averages = False
        for line in stdout.split('\n'):
            # only filter relevant lines that contain average metrics
            if "==== AVERAGE METRICS" in line:
                found_averages = True
            if not found_averages or "Average" not in line:
                continue
            parts = line.split(":")
            
            if len(parts) == 2:
                # extract metric name and value
                metric_name = parts[0].replace("Average ", "").strip()
                try:
                    value = float(parts[1].strip())
                    
                    # map metric names to be with -
                    if metric_name == "FMeasure": 
                        metrics["F-measure"] = value
                    elif metric_name == "PFMeasure": 
                        metrics["PF-measure"] = value
                    elif metric_name in metrics: 
                        metrics[metric_name] = value
                except ValueError:
                    continue
        return metrics, runtime
    except Exception as e:
        print(f"Error running evaluation for {model_name}: {str(e)}")
        return blank_metrics, None
    finally:
        os.chdir(current_dir)

def main():
    dataset_keys = []
    all_results = []
    
    if DEMO:
        dataset_keys = ["demo"]
    else:
        for k in get_all_dataset_keys():
            if k != "dibco2017":
                dataset_keys.append(k)

    for dataset_key in dataset_keys:
        print(f"Processing dataset: {dataset_key}\n")
        for model_name, config in MODELS.items():
            print(f"\nProcessing {model_name} on {dataset_key}...")
            # print(f"Model Directory: {config}")

            if EVAL_ONLY:
                runtime = None
                status = "Eval-Only"
            else:
                runtime, status = run_model(model_name, config, dataset_key)
                
            metrics, eval_runtime = run_evaluation(model_name, config, dataset_key)
            result_entry = {"Dataset": dataset_key, "Model": model_name, "Status": status}

            if runtime is not None:
                result_entry["Model Runtime (s)"] = f"{runtime:.2f}"
            else:
                result_entry["Model Runtime (s)"] = None

            if eval_runtime is not None:
                result_entry["Eval Runtime (s)"] = f"{eval_runtime:.2f}"
            else:
                result_entry["Eval Runtime (s)"] = None

            for k, v in metrics.items():
                if v is not None:
                    result_entry[k] = f"{v:.4f}"
                else:
                    result_entry[k] = None

            result_entry["Timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            all_results.append(result_entry)
            
    df = pd.DataFrame(all_results)
    df.to_csv(METRICS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    
    print("\n--- FINAL RESULTS ---")
    for dataset_key in dataset_keys:
        print(f"\n{dataset_key}:")
        print(df[df["Dataset"] == dataset_key].to_string(index=True))
    print(f"\nPipeline finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()