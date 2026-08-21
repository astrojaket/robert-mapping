from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_platform_environment_files_share_the_environment_name() -> None:
    for name in ("environment.yml", "environment-osx-arm64.yml", "environment-linux-64-cpu.yml"):
        payload = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
        assert payload["name"] == "eclipse-mapping"
        dependency_text = "\n".join(str(item) for item in payload["dependencies"])
        assert "jax" in dependency_text
        assert "numpyro" in dependency_text


def test_compute_profiles_set_three_cpu_threads() -> None:
    for name in ("laptop.env", "glamdring.env", "slurm.env"):
        text = (ROOT / "profiles" / name).read_text(encoding="utf-8")
        assert "OMP_NUM_THREADS=3" in text
        assert "XLA_FLAGS=\"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=3\"" in text

    batch = (ROOT / "profiles" / "slurm_fit.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in batch
    assert "#SBATCH --cpus-per-task=3" in batch


def test_example_configs_do_not_request_more_than_three_cpus() -> None:
    for path in sorted((ROOT / "examples").glob("*.yml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        compute = payload.get("compute", {})
        inference = payload.get("inference", {})
        assert compute.get("max_cpus", 3) <= 3, path
        assert compute.get("threads", 3) <= 3, path
        assert inference.get("chains", 3) <= 3, path
