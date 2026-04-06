# 🔹 1. FOR LOOP (Range)
print("----- FOR LOOP (Range) -----")

for i in range(1, 6):
    print("Number:", i)


# 🔹 2. FOR LOOP (List Iteration)
print("\n----- FOR LOOP (List) -----")

fruits = ["apple", "banana", "mango"]

for fruit in fruits:
    print("Fruit:", fruit)


# 🔹 3. FOR LOOP WITH CONDITION
print("\n----- FOR LOOP (Condition) -----")

numbers = [10, 25, 30, 45, 50]

for num in numbers:
    if num > 30:
        print(num, "is greater than 30")


# 🔹 4. WHILE LOOP (Basic)
print("\n----- WHILE LOOP -----")

count = 1

while count <= 5:
    print("Count:", count)
    count += 1   # IMPORTANT → prevents infinite loop


# 🔹 5. WHILE LOOP (User Control)
print("\n----- WHILE LOOP (User Control) -----")

num = 0

while num != -1:
    num = int(input("Enter a number (-1 to stop): "))
    print("You entered:", num)


# 🔹 6. BREAK STATEMENT
print("\n----- BREAK EXAMPLE -----")

for i in range(1, 10):
    if i == 5:
        print("Stopping loop at", i)
        break
    print(i)


# 🔹 7. CONTINUE STATEMENT
print("\n----- CONTINUE EXAMPLE -----")

for i in range(1, 6):
    if i == 3:
        continue
    print(i)


# 🔹 8. AVOIDING INFINITE LOOP (DEMO)
print("\n----- SAFE LOOP EXAMPLE -----")

x = 1

while x <= 3:
    print("Safe loop:", x)
    x += 1   # Without this → infinite loop


# 🔹 9. REAL-WORLD EXAMPLE
print("\n----- REAL-WORLD EXAMPLE -----")

marks = [85, 40, 78, 30, 90]

for mark in marks:
    if mark >= 50:
        print(mark, "Pass")
    else:
        print(mark, "Fail")