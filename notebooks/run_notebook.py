import papermill as pm
import sys
import argparse

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--notbeook_path', type=str, help="Path to the notebook file.", default="Baseline.ipynb")
    parser.add_argument('--cfg_path', type=str, help="Path to the cfg file.", default=None)
    pargs=parser.parse_args()

    result = pm.execute_notebook(
        input_path=pargs.notbeook_path,
        output_path="/dev/null",
        parameters={"cfg_path" : pargs.cfg_path},
        kernel_name='distmsmatch', 
        log_output=True,
        autosave_cell_every=60,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr,
    )

if __name__ == "__main__":
    main()
   