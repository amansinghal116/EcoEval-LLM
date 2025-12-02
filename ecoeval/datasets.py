# ecoeval/datasets.py
from typing import List
from datasets import Dataset


def _tiny_python_benchmark() -> Dataset:
    """
    Very small internal dataset of Python tasks with prompts + unit tests.
    Each row has: prompt, test_code
    """
    tasks = [
        {
            "prompt": "Write a Python function add(a, b) that returns their sum.",
            "test_code": "assert add(1, 2) == 3\nassert add(-1, 5) == 4",
        },
        {
            "prompt": "Write a Python function is_prime(n) that returns True if n is a prime number, False otherwise.",
            "test_code": (
                "assert is_prime(2)\n"
                "assert is_prime(3)\n"
                "assert not is_prime(4)\n"
                "assert is_prime(17)\n"
                "assert not is_prime(1)"
            ),
        },
        {
            "prompt": "Write a Python function factorial(n) that returns n! for non-negative n.",
            "test_code": (
                "assert factorial(0) == 1\n"
                "assert factorial(3) == 6\n"
                "assert factorial(5) == 120"
            ),
        },
        {
            "prompt": "Write a Python function reverse_string(s) that returns the reversed string.",
            "test_code": (
                "assert reverse_string('abc') == 'cba'\n"
                "assert reverse_string('') == ''\n"
                "assert reverse_string('racecar') == 'racecar'"
            ),
        },
        {
            "prompt": "Write a Python function fibonacci(n) that returns a list of the first n Fibonacci numbers, starting from 0.",
            "test_code": (
                "assert fibonacci(1) == [0]\n"
                "assert fibonacci(5) == [0, 1, 1, 2, 3]\n"
                "assert fibonacci(0) == []"
            ),
        },
        {
            "prompt": "Write a Python function count_vowels(s) that returns the number of vowels in the string s.",
            "test_code": (
                "assert count_vowels('hello') == 2\n"
                "assert count_vowels('xyz') == 0\n"
                "assert count_vowels('AEiou') == 5"
            ),
        },
        {
            "prompt": "Write a Python function flatten(lst) that takes a list of lists of integers and returns a single flattened list.",
            "test_code": (
                "assert flatten([[1,2],[3,4]]) == [1,2,3,4]\n"
                "assert flatten([]) == []\n"
                "assert flatten([[1],[2],[3]]) == [1,2,3]"
            ),
        },
        {
            "prompt": "Write a Python function is_palindrome(s) that returns True if s is a palindrome, ignoring case and spaces.",
            "test_code": (
                "assert is_palindrome('racecar')\n"
                "assert is_palindrome('RaceCar')\n"
                "assert is_palindrome('nurses run')\n"
                "assert not is_palindrome('hello')"
            ),
        },
    ]
    return Dataset.from_list(tasks)


_AVAILABLE_DATASETS = {
    "tiny-python-benchmark": _tiny_python_benchmark,
}


def list_available_datasets() -> List[str]:
    return sorted(_AVAILABLE_DATASETS.keys())


def load_dataset_by_name(name: str) -> Dataset:
    if name not in _AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    return _AVAILABLE_DATASETS[name]()
