"""Exit 0 if the dream sidecar's heartbeat is fresh, 1 otherwise.

Used as the docker-compose healthcheck for the `dreamer` service.
"""

import sys

from src.memory.dream import heartbeat_age_seconds
from src.utils.config import config


def main() -> int:
    age = heartbeat_age_seconds(config.results_db_path)
    limit = 3 * max(config.memory.dream.interval_seconds, 60)
    if age is None:
        print("no dream heartbeat yet")
        return 1
    print(f"last dream cycle {age:.0f}s ago (limit {limit}s)")
    return 0 if age <= limit else 1


if __name__ == "__main__":
    sys.exit(main())
