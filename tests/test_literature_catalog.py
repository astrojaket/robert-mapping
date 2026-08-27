from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "literature_data" / "catalog.yml"


def test_literature_catalog_has_unique_dataset_ids_and_safe_policy() -> None:
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    datasets = catalog["datasets"]
    ids = [item["id"] for item in datasets]

    assert len(ids) == len(set(ids))
    assert catalog["policy"]["max_parallel_downloads"] == 3
    assert catalog["policy"]["run_inference"] is False
    assert catalog["policy"]["run_simulations"] is False
    assert len(datasets) >= 20
    assert "download_queued" not in {item["state"] for item in datasets}


def test_direct_downloads_use_named_planet_and_instrument_folders() -> None:
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))

    for dataset in catalog["datasets"]:
        directory = Path(dataset["directory"])
        assert len(directory.parts) == 2
        assert dataset["planet"].lower().replace(" ", "") in (
            directory.parts[0].lower().replace("-", ""),
            directory.parts[0].lower().replace("-", "").replace("cancrie", "cnce"),
        ) or dataset["id"].startswith(directory.parts[0].lower().replace("-", ""))
        for item in dataset.get("files", []):
            assert item["name"]
            assert item["url"].startswith("https://")
