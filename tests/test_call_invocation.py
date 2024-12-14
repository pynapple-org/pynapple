import ast
import importlib
import inspect
import pkgutil
import sys
import textwrap

import pytest
from numba import jit

import pynapple as nap


def valid_func(x):
    # do not call
    x.__class__


@jit
def valid_func_decorated(x):
    x.__class__


def invalid_func(x):
    x.__class__()


@jit
def invalid_func_decorated(x):
    x.__class__()


class BaseClass:
    def method(self):
        pass


class ValidClass(BaseClass):
    def __init__(self):
        pass

    def method(self):
        self.__class__


class InvalidClass(BaseClass):
    def __init__(self):
        pass

    def method(self):
        self.__class__()


class ValidClassNoInheritance:
    def __init__(self):
        pass

    def method(self):
        self.__class__()


def is_function_or_wrapped_function(obj):
    """
    Custom predicate to identify functions, including those wrapped by decorators.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is a function or a wrapped function.
    """
    # Unwrap the object if it’s wrapped by decorators
    unwrapped = inspect.unwrap(
        obj, stop=(lambda f: inspect.isfunction(f) or inspect.isbuiltin(f))
    )
    return inspect.isfunction(unwrapped)


def class_method_invocations(cls, method_name):
    class_results = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        try:
            # Get the source code of the method
            source = textwrap.dedent(inspect.getsource(method))
            # Parse the source into an abstract syntax tree
            tree = ast.parse(source)
            # Walk the AST to check for `__call__` invocations
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == method_name:
                        class_results.append(name)
                        break
        except Exception as e:
            # cannot grab source code of inherited methods.
            print(cls, name, method, repr(e))
            pass
    try:
        class_results.remove("_define_instance")
    except ValueError:
        pass
    try:
        class_results.remove("_initialize_tsd_output")
    except ValueError:
        pass
    return class_results


def subclass_method_invocations(base_class, method_name):
    """
    Finds methods in subclasses of a base class where the `method_name` method is invoked.

    Args:
        base_class (type): The base class to inspect.
        method_name: string

    Returns:
        dict: A dictionary with subclass names as keys and a list of method names invoking `method_name`.
    """
    results = {}

    cls_results = class_method_invocations(base_class, method_name)

    if cls_results:
        results[base_class.__name__] = cls_results

    for subclass in base_class.__subclasses__():

        subclass_results = class_method_invocations(subclass, method_name)
        if subclass_results:
            results[subclass.__name__] = subclass_results

    return results


def find_method_invocations_in_function(func, method_name):
    """
    Checks if a function contains a call to `method_name`.

    Parameters
    ----------
        func (callable): The function to analyze.
        method_name: the name of the method

    Returns
    -------
        bool: True if `method_name` is invoked in the function, False otherwise.
    """
    try:
        # Get the source code of the function
        source = textwrap.dedent(inspect.getsource(func))
        # Parse the source into an abstract syntax tree
        tree = ast.parse(source)
        # Walk the AST to check for `__call__` invocations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(
                node.func, (ast.Attribute, ast.Name)
            ):
                name = getattr(node.func, "attr", getattr(node.func, "id", None))
                if name == method_name:
                    return True
    except Exception as e:
        # Log the function that couldn't be analyzed
        print(f"Could not analyze function {func}: {e}")
    return False


def find_method_invocations_in_module_functions(module, method_name):
    """
    Recursively find all functions in a module that invoke `method_name`.

    Parameters
    ----------
    module (module): The module to inspect.
    method_name (str): The name of the method to inspect.

    Returns
    -------
    dict: A dictionary with module, class, or function names as keys and
        a list of function/method names invoking `method_name`.
    """
    results_func = {}

    # Inspect functions directly defined in the module
    for name, func in inspect.getmembers(
        module, predicate=is_function_or_wrapped_function
    ):
        if find_method_invocations_in_function(func, method_name):
            results_func[module.__name__ + f".{name}"] = name

    # Recursively inspect submodules
    if hasattr(module, "__path__"):  # Only packages have a __path__
        for submodule_info in pkgutil.iter_modules(module.__path__):
            submodule_name = f"{module.__name__}.{submodule_info.name}"
            submodule = importlib.import_module(submodule_name)
            submodule_results = find_method_invocations_in_module_functions(
                submodule, method_name
            )
            if submodule_results:
                results_func.update(submodule_results)

    return results_func


def test_find_func():
    # Get the current module
    current_module = sys.modules[__name__]

    # Run the detection function
    results = find_method_invocations_in_module_functions(current_module, "__class__")
    expected_results = {
        "tests.test_call_invocation.invalid_func": "invalid_func",
        "tests.test_call_invocation.invalid_func_decorated": "invalid_func_decorated",
    }
    assert results == expected_results


def test_find_class():
    # Run the detection function
    results = subclass_method_invocations(BaseClass, "__class__")
    expected_results = {"InvalidClass": ["method"]}
    assert results == expected_results


def test_no_direct__class__invocation_in_base_subclasses():
    results_func = find_method_invocations_in_module_functions(nap, "__class__")
    results_cls = subclass_method_invocations(nap.core.base_class._Base, "__class__")
    if results_cls != {}:
        raise ValueError(
            f"Direct use of __class__ found in the following _Base objects and methods: {results_cls}. \n"
            "Please, replace them with `_define_instance` or `_initialize_tsd_output`."
        )

    if results_cls != {}:
        raise ValueError(
            f"Direct use of __class__ found in the following modules and functions: {results_func}. \n"
            "Please, replace them with `_define_instance` or `_initialize_tsd_output`."
        )


def test_no_direct_get_cls_invocation_in_base_subclasses():
    results_func = find_method_invocations_in_module_functions(nap, "_get_class")
    results_cls = subclass_method_invocations(nap.core.base_class._Base, "_get_class")
    if results_cls != {}:
        raise ValueError(
            f"Direct use of `_get_cls` found in the following _Base objects and methods: {results_cls}. \n"
            "Please, replace them with `_initialize_tsd_output`."
        )

    if results_func != {
        "pynapple.core.time_series._initialize_tsd_output": "_initialize_tsd_output"
    }:
        raise ValueError(
            f"Direct use of _get_cls found in the following modules and functions: {results_func}. \n"
            "Please, replace them with `_initialize_tsd_output`."
        )
