# src/basics/numpy_structure.py

import numpy as np


def create_1d_array():
    """
    Create a 1D NumPy array.
    """
    one_d_array = np.array([10, 20, 30, 40])
    return one_d_array


def create_2d_array():
    """
    Create a 2D NumPy array.
    """
    two_d_array = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    return two_d_array


def inspect_array_properties(array):
    """
    Print shape and dimensions of the array.
    """
    print("Array:\n", array)
    print("Shape:", array.shape)
    print("Dimensions (ndim):", array.ndim)


def access_1d_elements(array):
    """
    Access elements in a 1D array using index positions.
    """
    print("\nAccessing 1D Array Elements:")
    print("First element (index 0):", array[0])
    print("Second element (index 1):", array[1])
    print("Last element (index -1):", array[-1])


def access_2d_elements(array):
    """
    Access elements in a 2D array using row and column indices.
    """
    print("\nAccessing 2D Array Elements:")

    # Row 0, Column 0
    print("Element at (0, 0):", array[0, 0])

    # Row 0, Column 2
    print("Element at (0, 2):", array[0, 2])

    # Row 1, Column 1
    print("Element at (1, 1):", array[1, 1])


def visualize_array_layout(array):
    """
    Explain how data is arranged in a 2D array.
    """
    print("\nVisualizing 2D Array Layout:")
    print("Row 0:", array[0])
    print("Row 1:", array[1])

    print("Column 0:", array[:, 0])
    print("Column 1:", array[:, 1])


def main():
    """
    Main execution function.
    """

    # 1D Array
    print("----- 1D ARRAY -----")
    one_d_array = create_1d_array()
    inspect_array_properties(one_d_array)
    access_1d_elements(one_d_array)

    # 2D Array
    print("\n----- 2D ARRAY -----")
    two_d_array = create_2d_array()
    inspect_array_properties(two_d_array)
    access_2d_elements(two_d_array)
    visualize_array_layout(two_d_array)


if __name__ == "__main__":
    main()