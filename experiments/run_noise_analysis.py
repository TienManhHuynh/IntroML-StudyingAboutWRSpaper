"""Run matched noise rates for one learner/dataset; test labels remain clean."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_reproduction import parser, run_experiment


def main():
    """Run each requested train-noise probability into a distinct output directory."""
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=False)
    p = parser()
    p.add_argument('--rates', nargs='+', type=float, default=[0,.05,.1,.2,.3])
    config = vars(p.parse_args())
    output, rates = config.pop('output'), config.pop('rates')
    if len(set(rates)) != len(rates) or any(not 0<=rate<=1 for rate in rates):
        p.error('Rates must be unique and lie in [0,1]')
    # Every run reuses the dataset/split seed and noise RNG seed: paired nested masks.
    for rate in rates:
        trial = dict(config, noise_rate=rate, experiment='label-noise-development' if config['dataset']=='synthetic' or config['max_samples'] else 'label-noise')
        print(json.dumps(run_experiment(trial, output/('noise-'+format(rate,'.8g')))))


if __name__ == '__main__':
    main()
