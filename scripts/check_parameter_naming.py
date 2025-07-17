import logging
import sys
import pynapple as nap  # keep this import at the top if you prefer

logger = logging.getLogger("check_parameter_naming")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    params = collect_similar_parameter_names(nap, similarity_cutoff=0.9)

    # Remove non-conflicting parameter names
    for name, occurrences in list(params.items()):
        if len(occurrences) == 1:
            params.pop(name)

    if params:
        msg_lines = ["Inconsistency in parameter naming found!\n"]
        for name, occurrences in params.items():
            msg_lines.append(f"{name}:\n")
            for param_name, path in occurrences:
                msg_lines.append(f"\t- {path}: {param_name}\n")
            msg_lines.append("\n")

        logger.error("".join(msg_lines))
        sys.exit(1)
    else:
        logger.info("No parameter naming inconsistencies found.")
