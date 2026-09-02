import os


def test_issue_on_upstream_fail():
    # Intentionally fails on upstream workflow run
    job_name = os.environ.get("GITHUB_JOB")

    if job_name is None:
        return  # not running in CI

    if job_name != "upstream-dev":
        return  # not the upstream workflow

    assert 1 == 2, "Failing out to test the upstream workflow"
