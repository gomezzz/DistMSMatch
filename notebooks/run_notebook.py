import papermill as pm
import sys
import argparse

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook_path', type=str, help="Path to the notebook file.", default="Baseline.ipynb")
    parser.add_argument('--cfg_path', type=str, help="Path to the cfg file.", default=None)
    parser.add_argument('--dataset', type=str, help="Dataset used for simulations.", default="eurosat_rgb")
    parser.add_argument('--simulation_path', type=str, help="Path to simulation results.", default="saved_models/test")
    pargs=parser.parse_args()

    result = pm.execute_notebook(
        input_path=pargs.notebook_path,
        output_path="/dev/null",
        parameters={"cfg_path" : pargs.cfg_path, "dataset" : pargs.dataset, "simulation_path" : pargs.simulation_path},
        kernel_name='distmsmatch', 
        log_output=True,
        autosave_cell_every=60,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr,
    )

if __name__ == "__main__":
    main()
   