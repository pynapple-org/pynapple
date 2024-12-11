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


def class_class_invocations(cls):
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
                    if node.func.attr == "__class__":
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
    return class_results


def subclass_class_invocations(base_class):
    """
    Finds methods in subclasses of a base class where the `__class__` method is invoked.

    Args:
        base_class (type): The base class to inspect.

    Returns:
        dict: A dictionary with subclass names as keys and a list of method names invoking `__class__`.
    """
    results = {}

    cls_results = class_class_invocations(base_class)

    if cls_results:
        results[base_class.__name__] = cls_results

    for subclass in base_class.__subclasses__():

        subclass_results = class_class_invocations(subclass)
        if subclass_results:
            results[subclass.__name__] = subclass_results

    return results


def find_class_invocations_in_function(func):
    """
    Checks if a function contains a call to `__class__`.

    Args:
        func (callable): The function to analyze.

    Returns:
        bool: True if `__class__` is invoked in the function, False otherwise.
    """
    try:
        # Get the source code of the function
        source = textwrap.dedent(inspect.getsource(func))
        # Parse the source into an abstract syntax tree
        tree = ast.parse(source)
        # Walk the AST to check for `__call__` invocations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "__class__":
                    return True
    except Exception as e:
        # Log the function that couldn't be analyzed
        print(f"Could not analyze function {func}: {e}")
    return False


def find_class_invocations_in_module_functions(module):
    """
    Recursively find all functions in a module that invoke `__class__`.

    Args:
        module (module): The module to inspect.

    Returns:
        dict: A dictionary with module, class, or function names as keys and
              a list of function/method names invoking `__class__`.
    """
    results_func = {}

    # Inspect functions directly defined in the module
    for name, func in inspect.getmembers(
        module, predicate=is_function_or_wrapped_function
    ):
        if find_class_invocations_in_function(func):
            results_func[module.__name__ + f".{name}"] = name

    # Recursively inspect submodules
    if hasattr(module, "__path__"):  # Only packages have a __path__
        for submodule_info in pkgutil.iter_modules(module.__path__):
            submodule_name = f"{module.__name__}.{submodule_info.name}"
            submodule = importlib.import_module(submodule_name)
            submodule_results = find_class_invocations_in_module_functions(submodule)
            if submodule_results:
                results_func.update(submodule_results)

    return results_func


def test_find_func():
    # Get the current module
    current_module = sys.modules[__name__]

    # Run the detection function
    results = find_class_invocations_in_module_functions(current_module)
    expected_results = {
        "tests.test_call_invocation.invalid_func": "invalid_func",
        "tests.test_call_invocation.invalid_func_decorated": "invalid_func_decorated",
    }
    assert results == expected_results


def test_find_class():
    # Run the detection function
    results = subclass_class_invocations(BaseClass)
    expected_results = {"InvalidClass": ["method"]}
    assert results == expected_results


def test_no_direct__class__invocation_in_base_subclasses():
    results_func = find_class_invocations_in_module_functions(nap)
    results_cls = subclass_class_invocations(nap.core.base_class._Base)
    if results_cls != {}:
        raise ValueError(
            f"Direct use of __class__ found in the following _Base objects and methods: {results_cls}. \n"
            "Please, replace them with `_define_instance`."
        )

    if results_cls != {}:
        raise ValueError(
            f"Direct use of __class__ found in the following modules and functions: {results_func}. \n"
            "Please, replace them with `_define_instance`."
        )
