import ast
import difflib
import itertools
import os
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional

# Pairs of parameter names that are lexically similar but intentionally allowed.

# During parameter name similarity checks, some pairs of names may be flagged
# as potentially inconsistent due to their high string similarity. This list
# enumerates such known, acceptable pairs that should be *excluded* from warnings.

# Each pair is stored as a set of two strings (e.g., {"a", "a_1"}), and comparison
# is done using set equality, i.e., order does not matter.

# These typically include:
# - semantically equivalent alternatives (e.g., {"conv_time_series", "time_series"})
# - mirrored structures (e.g., {"inhib_a", "inhib_b"})
# - systematic naming conventions (e.g., {"basis1", "basis2"})
# - commonly used argument patterns (e.g., {"args", "kwargs"})
VALID_PAIRS = [
    {"ep", "sep"},
    {"ts", "tsd"},
    {"args", "kwargs"},
    {"channel", "n_channels"},
    {"interval_size", "intervalset"},
    {"new_time_support", "time_support"},
    {"ufunc", "func"},
    {"keys", "key"},
    {"value", "values"},
    *({a, b} for (a, b) in itertools.combinations(["starts", "start", "start1", "start2"], r=2)),
    *({a, b} for (a, b) in itertools.combinations(["ends", "end", "end1", "end2"], r=2)),
    {"windowsize", "window"},
    {"windowsize", "windows"},
]


def handle_matches(
    current_parameter: str,
    current_path: str,
    matches: List[str],
    results: Dict,
    valid_pairs: List[set[str]],
):
    """
    Handle matched parameter names by updating or creating groups in the results dictionary.

    A parameter is considered valid if it has no matches or if all its matches appear in
    `valid_pairs` as a set with the current parameter. Valid parameters are added as new entries
    in the results dictionary. Invalid parameters (i.e., those with partial or conflicting matches)
    are added to existing groups if any of their matches are already present in those groups.

    Note: This function allows overlapping groups. If `current_parameter` is similar to multiple
    parameter groups (e.g., "timin" may match both "time" and "timing"), it will be added to each
    of the matching groups independently.

    Parameters
    ----------
    current_parameter :
        The name of the parameter currently being processed.

    current_path :
        The path or context in which the parameter was found (e.g., a file or data structure path).

    matches :
        A list of other parameter names that are similar to ``current_parameter``.

    results :
        A dictionary of grouped parameters. Keys are group names, and values are dictionaries
        containing:
            - "unique_names": a set of parameter names in the group.
            - "info": a list of (parameter, path) tuples for matched entries.

    valid_pairs :
        A list of valid two-element sets. Each set contains a pair of parameter names that are
        considered equivalent or compatible.

    """
    # a parameter name is valid if no matches or all matches in valid pairs
    list_invalid = [
        match for match in matches if {match, current_parameter} not in valid_pairs
    ]
    if len(list_invalid) == 0:
        # if all matches are valid, create a new group for this parameter
        results[current_parameter] = {
            "unique_names": {current_parameter},
            "info": [(current_parameter, current_path)],
        }
    else:

        # if there is an invalid match, then add to existing result entry
        for k, v in results.items():
            # Otherwise, add the parameter to any existing groups where it has a match
            #
            # Note: We *intentionally allow overlapping groups*. If `current_parameter`
            # is similar to multiple different parameter groups
            # (e.g. "timin" may be similar to both "time" and "timing", but "time" and "timing" may
            # belong to two different groups),
            # it will be added to each of those groups.
            is_in_category = any(match in v["unique_names"] for match in list_invalid)
            if is_in_category:
                v["info"].append((current_parameter, current_path))
                v["unique_names"].add(current_parameter)


def extract_parameters_from_ast(
    tree: ast.Module,
    file_path: pathlib.Path,
    results: Dict,
    valid_pairs: List[set[str]],
    unique_param_names: set,
    similarity_cutoff: float,
):

    class ParamVisitor(ast.NodeVisitor):
        def __init__(self):
            self.class_name = None

        def visit_ClassDef(self, node):
            prev_class = self.class_name
            self.class_name = node.name
            self.generic_visit(node)
            self.class_name = prev_class

        def visit_FunctionDef(self, node):
            qualified_name = (
                f"{self.class_name}.{node.name}" if self.class_name else node.name
            )
            param_names = [str(arg.arg) for arg in node.args.args]
            for par in param_names:
                # if perfect match is present just add there
                if par in results:
                    results[par]["unique_names"].add(par)
                    results[par]["info"].append(
                        (par, f"{file_path.as_posix()}:{qualified_name}")
                    )
                    continue

                matches = difflib.get_close_matches(
                    par, unique_param_names, n=100, cutoff=similarity_cutoff
                )
                handle_matches(
                    par,
                    f"{file_path.as_posix()}:{qualified_name}",
                    matches,
                    results,
                    valid_pairs,
                )
                unique_param_names.add(par)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

    ParamVisitor().visit(tree)


def collect_similar_parameter_names_ast(
    root_dir: str | pathlib.Path,
    similarity_cutoff: float = 0.8,
    valid_pairs: Optional[List[set[str]]] = None,
) -> Dict[str, Dict]:
    if valid_pairs is None:
        valid_pairs = VALID_PAIRS

    results = {}
    unique_param_names = set()

    for dirpath, _, filenames in os.walk(root_dir):
        dirpath = pathlib.Path(dirpath)

        if "third_party" in dirpath.parts:
            continue

        for filename in filenames:
            if filename.endswith(".py"):
                full_path = dirpath / filename
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source, filename=full_path)
                    extract_parameters_from_ast(
                        tree,
                        full_path,
                        results,
                        valid_pairs,
                        unique_param_names,
                        similarity_cutoff,
                    )
                except (UnicodeDecodeError, FileNotFoundError):
                    continue

    return results


if __name__ == "__main__":
    import argparse
    import logging
    import sys

    default_path = pathlib.Path(__file__).parent.parent / "pynapple"

    parser = argparse.ArgumentParser(
        description="Check parameter naming consistency using AST."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=pathlib.Path,
        help="Root path to the package (source folder).",
        default=default_path,
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Similarity threshold for parameter name grouping.",
    )
    args = parser.parse_args()

    logger = logging.getLogger("check_parameter_naming")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    params = collect_similar_parameter_names_ast(
        args.path, similarity_cutoff=args.threshold
    )
    invalid = [name for name, d in params.items() if len(d["unique_names"]) > 1]

    if invalid:
        msg_lines = ["Inconsistency in parameter naming found!\n"]
        for name in invalid:
            msg_lines.append(f"{name}:\n")
            grouped_info = defaultdict(list)
            for param_name, path in sorted(params[name]["info"], key=lambda x: x[1]):
                grouped_info[param_name].append(path)
            for param_name in sorted(params[name]["unique_names"]):
                msg_lines.append(f"\t- {param_name}:\n")
                for path in grouped_info[param_name]:
                    msg_lines.append(f"\t\t- {path}\n")
            msg_lines.append("\n")
        logger.error("".join(msg_lines))
        sys.exit(1)
    else:
        logger.info("No parameter naming inconsistencies found.")
