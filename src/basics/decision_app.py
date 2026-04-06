# 🔹 SMART STUDENT EVALUATION SYSTEM

print("===== STUDENT EVALUATION SYSTEM =====")

name = input("Enter student name: ")
marks = int(input("Enter marks: "))
attendance = int(input("Enter attendance (%): "))

# Basic pass/fail
if marks >= 50:
    result = "Pass"
else:
    result = "Fail"

print(f"{name} Result:", result)


# 🔹 Grade System (if-elif-else)
if marks >= 90:
    grade = "A"
elif marks >= 75:
    grade = "B"
elif marks >= 60:
    grade = "C"
elif marks >= 50:
    grade = "D"
else:
    grade = "F"

print("Grade:", grade)


# 🔹 Scholarship Eligibility (Logical Operators)
if marks >= 85 and attendance >= 80:
    print("Eligible for Scholarship")
else:
    print("Not eligible for Scholarship")


# 🔹 Warning System (using NOT)
if not (attendance >= 75):
    print("⚠️ Low attendance warning!")


# 🔹 Special Message (String Condition)
if name.lower() == "vijayashree":
    print("⭐ Top Performer Detected!")


# 🔹 Final Summary
print("\n===== FINAL SUMMARY =====")
print(f"Name: {name}")
print(f"Marks: {marks}")
print(f"Attendance: {attendance}%")
print(f"Grade: {grade}")