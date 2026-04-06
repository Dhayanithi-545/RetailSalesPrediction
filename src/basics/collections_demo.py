# 🔹 LIST EXAMPLE
print("----- LIST EXAMPLE -----")
numbers = [10, 20, 30, 40]

print("Original list:", numbers)

# Access
print("First element:", numbers[0])

# Modify
numbers[1] = 25

# Add
numbers.append(50)

# Remove
numbers.remove(30)

print("Updated list:", numbers)

# Iterate
print("Iterating list:")
for num in numbers:
    print(num)


# 🔹 TUPLE EXAMPLE
print("\n----- TUPLE EXAMPLE -----")
data = (1, 2, 3, 4)

print("Tuple:", data)
print("First element:", data[0])

# Uncomment to show error in video
# data[1] = 10  # ❌ This will cause error


# 🔹 DICTIONARY EXAMPLE
print("\n----- DICTIONARY EXAMPLE -----")
student = {
    "name": "Vijayashree",
    "age": 20,
    "course": "Python"
}

print("Original dictionary:", student)

# Access
print("Name:", student["name"])

# Modify
student["age"] = 21

# Add
student["grade"] = "A"

# Remove
del student["course"]

print("Updated dictionary:", student)


# 🔹 DIFFERENCE DEMO
print("\n----- DIFFERENCE DEMO -----")

# List (mutable)
my_list = [1, 2, 3]
my_list[0] = 100
print("List changed:", my_list)

# Tuple (immutable)
my_tuple = (1, 2, 3)
print("Tuple (unchanged):", my_tuple)

# Dictionary (mutable)
my_dict = {"a": 1}
my_dict["a"] = 99
print("Dictionary changed:", my_dict)


# 🔹 REAL-WORLD EXAMPLE
print("\n----- REAL-WORLD EXAMPLE -----")

marks = [85, 90, 78]  # list
coordinates = (10.5, 20.3)  # tuple
person = {"name": "Vijayashree", "age": 20}  # dictionary

print("Marks:", marks)
print("Coordinates:", coordinates)
print("Person:", person)