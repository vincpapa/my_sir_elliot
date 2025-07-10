from elliot.run import run_experiment
import glob
import os

# parser = argparse.ArgumentParser(description="Run sample main.")
# parser.add_argument('--config', type=str, default='svdgcn')
# args = parser.parse_args()
per_user = False
conf_list = glob.glob(os.sep.join(["config_files/amazon_lightgcn.yml"]))
# conf_list = ["config_files/ml1m_lightgcn_6.yml", "config_files/ml1m_lightgcn_7.yml"]
for conf_file in conf_list:
    run_experiment(f"{conf_file}", per_user)

