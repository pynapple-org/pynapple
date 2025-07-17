import difflib
import inspect
import types
import logging
import sys
import pynapple as nap


def collect_similar_parameter_names(package, root_name=None, similarity_cutoff=0.8):
    """
    Recursively collect and group similar parameter names from functions and methods.

    This function traverses the given package and its submodules, extracting parameter
    names from all user-defined functions and methods. Parameter names that are
    lexically similar (based on `difflib.get_close_matches`) are grouped together.
    This can be used to detect inconsistent naming conventions across a codebase.

    Parameters
    ----------
    package : module
        The root package to analyze (e.g., `pynapple`).
    root_name : str, optional
        The dotted name of the root package. If not provided, it is inferred from
        `package.__name__`.
    similarity_cutoff : float, optional
        Similarity threshold between 0 and 1 used to group parameters based on
        lexical similarity (default is 0.8).

    Returns
    -------
    dict
        A dictionary mapping canonical parameter names to a list of tuples.
        Each tuple contains:
            - The actual parameter name
            - The fully qualified function or method path where it appears

        Example
        -------
        {
            "time": [("time", "pynapple.core.Tsd.__init__"), ("t", "pynapple.io.load")],
            ...
        }
    """
    if root_name is None:
        root_name = package.__name__

    results = {}
    visited_ids = set()

    def process_function(func, path):
        try:
            sig = inspect.signature(func)
            param_names = list(sig.parameters)
            for par in param_names:
                if par in results:
                    continue  # exact name already exists
                match = difflib.get_close_matches(par, results.keys(), n=1, cutoff=similarity_cutoff)
                if match:
                    results[match[0]].append((par, path))
                else:
                    results[par] = [(par, path)]
        except Exception:
            pass  # some built-ins or extension modules may not support signature()

    def walk(obj, path_prefix=""):
        if id(obj) in visited_ids:
            return
        visited_ids.add(id(obj))

        if inspect.isfunction(obj) or inspect.ismethod(obj):
            if getattr(obj, '__module__', '').startswith(root_name):
                process_function(obj, path_prefix)

        elif inspect.isclass(obj):
            if getattr(obj, '__module__', '').startswith(root_name):
                for name, member in inspect.getmembers(obj):
                    if name.startswith("_"):
                        continue
                    walk(member, f"{path_prefix}.{name}")

        elif isinstance(obj, types.ModuleType):
            if not getattr(obj, '__name__', '').startswith(root_name):
                return  # external module, skip
            for name, member in inspect.getmembers(obj):
                if name.startswith("_"):
                    continue
                walk(member, f"{path_prefix}.{name}")

    walk(package, package.__name__)
    return results


if __name__ == "__main__":
    logger = logging.getLogger("check_parameter_naming")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    params = collect_similar_parameter_names(nap, similarity_cutoff=0.9)

    # Filter out parameter names that occur only once
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
