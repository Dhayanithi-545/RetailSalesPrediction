# src/basics/numpy_demo.py

import numpy as np


def create_1d_array():
    """
    Create a 1D NumPy array from a Python list.
    """
    sales_list = [10, 20, 30, 40, 50]
    sales_array = np.array(sales_list)

    return sales_array


def create_2d_array():
    """
    Create a 2D NumPy array from nested Python lists.
    """
    weekly_sales_list = [
        [10, 20, 30],
        [40, 50, 60]
    ]

    weekly_sales_array = np.array(weekly_sales_list)

    return weekly_sales_array


def inspect_array(array):
    """
    Print array properties like shape, dimensions, and data type.
    """
    print("Array:\n", array)
    print("Shape:", array.shape)
    print("Dimensions:", array.ndim)
    print("Data Type:", array.dtype)


def perform_array_operations(array):
    """
    Perform basic arithmetic operations on NumPy array.
    """
    print("\nOriginal Array:", array)

    print("Array + 10:", array + 10)
    print("Array * 2:", array * 2)
    print("Array Mean:", np.mean(array))


def compare_list_vs_array():
    """
    Demonstrate difference between Python list and NumPy array operations.
    """
    python_list = [1, 2, 3]
    numpy_array = np.array(python_list)

    print("\nPython List + Python List:", python_list + python_list)
    print("NumPy Array + NumPy Array:", numpy_array + numpy_array)


def main():
    """
    Main execution function.
    """
    # Create arrays
    one_d_array = create_1d_array()
    two_d_array = create_2d_array()

    # Inspect arrays
    print("----- 1D Array -----")
    inspect_array(one_d_array)

    print("\n----- 2D Array -----")
    inspect_array(two_d_array)

    # Perform operations
    perform_array_operations(one_d_array)

    # Compare behavior
    compare_list_vs_array()


if __name__ == "__main__":
    main()