"""Command line of main.py: common options + one --<key> override per configuration key."""
import argparse

from utility.utility_function.tools import parse_value

STUDIES = ("dev", "convergence", "pilot", "controls", "comparison", "hierarchy", "cold_start", "transfer", "ablation")


def _base_parser(add_help=True):
    parser = argparse.ArgumentParser(description="HiTaxProto-CF runner", add_help=add_help)
    parser.add_argument("--model", required=True, help="configure/<model>.txt (LightGCN | HiTaxProtoCF)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="training seeds, run sequentially")
    parser.add_argument("--study", default="dev", choices=STUDIES, help="experiment family (docs/GLOSSARY.md)")
    parser.add_argument("--setting", default=None,
                        help="setting folder name (default: built from the key hyperparameters of the model)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--reference_overall", type=float, default=None,
                        help="one validation Overall Recall@20 of the reference LightGCN, used for every seed")
    parser.add_argument("--reference_selection", default=None,
                        help="selection report of the reference LightGCN (file name under results/selection/): each "
                             "seed uses the validation Overall Recall@20 of the reference run of the SAME seed")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="epochs between resumable checkpoints")
    parser.add_argument("--resume", action="store_true", help="resume from last_model.pt of the same run directory")
    parser.add_argument("--validation_users", choices=("all", "selection"), default="all",
                        help="selection: early stopping on the selection half only; the assessment half is evaluated "
                             "once on the selected checkpoint (pilot)")
    parser.add_argument("--protocol", default=None,
                        help="frozen protocol file; required to run on the locked confirmation dataset")
    return parser


def parse_args(config, argv=None):
    parser = _base_parser()
    for key in sorted(config):
        parser.add_argument("--" + key, dest="config_" + key, default=None, metavar="VALUE",
                            help="override configuration key {}".format(key))
    return parser.parse_args(argv)


def parse_model(argv=None):
    args, _ = _base_parser(add_help=False).parse_known_args(argv)
    return args.model


def overrides(config, args):
    """{key: (new, default)} for every key given on the command line."""
    out = {}
    for key in config:
        value = getattr(args, "config_" + key, None)
        if value is not None:
            out[key] = (parse_value(value), config[key])
    return out


def apply_overrides(config, args):
    out = dict(config)
    for key, (value, _) in overrides(config, args).items():
        out[key] = value
    return out
