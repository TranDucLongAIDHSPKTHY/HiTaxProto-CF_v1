"""Configuration files: typed parsing of configure/*.txt and stable hashing."""
import pytest

from config_path.config_path import model_config_file
from utility.utility_function.tools import config_hash, parse_value, read_configuration


def test_parse_value_types():
    assert parse_value(" none ") is None and parse_value("true") is True and parse_value("False") is False
    assert parse_value("2048") == 2048 and parse_value("0.0001") == 1e-4 and parse_value("[20, 10]") == [20, 10]
    assert parse_value("amazon-book") == "amazon-book" and parse_value("taxonomy") == "taxonomy"


@pytest.mark.parametrize("model", ["LightGCN", "HiTaxProtoCF"])
def test_shipped_configurations_parse(model):
    config = read_configuration(model_config_file(model))
    assert config["K"] == 3 and config["batch_size"] == 2048 and config["ks"] == [10, 20]
    assert config_hash(config) == config_hash(dict(reversed(list(config.items()))))


def test_duplicate_key_rejected(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("lr = 0.001\nlr = 0.002\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_configuration(path)
