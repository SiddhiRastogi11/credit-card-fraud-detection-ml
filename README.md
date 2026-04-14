Credit Card Validator
A robust Python utility designed to verify the integrity of credit card numbers. This project implements defensive programming patterns to handle real-world input scenarios, ensuring data accuracy before any backend processing occurs.

 Project Objective
The goal was to build a reliable validation layer that sanitizes raw user input and applies rule-based logic to confirm card authenticity, preventing malformed data from reaching the database.

 Key Features
Rule-Based Logic: Implements format checks including length verification (13-16 digits) and major industry identifier (MII) detection.

Input Sanitization: Utilizes string manipulation to strip whitespace, hyphens, and non-numeric characters.

Defensive Programming: Integrated comprehensive try-except blocks and conditional checks to handle edge cases like empty strings or "null" inputs.

Modular Architecture: The logic is encapsulated in reusable functions, making it easy to import into a larger Spring Boot or Flask payment gateway.

 Tech Stack
Language: Python 3.x

Libraries: re (Regular Expressions) for pattern matching.

Testing: Manual test cases covering valid, invalid, and edge-case scenarios.

 How It Works
Normalization: The system receives a string and removes all non-digit characters.

Validation: The sanitized string is checked against specific length and prefix rules (e.g., Visa starts with 4, Mastercard starts with 5).

Result: Returns a boolean or a descriptive error message indicating why the card failed validation.

 Sample Usage
Python
from validator import validate_card

# Test a valid card string with hyphens
result = validate_card("4111-1111-1111-1111")
print(result) # Output: True
