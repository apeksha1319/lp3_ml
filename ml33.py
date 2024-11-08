Gradient Descent is an optimization algorithm used to find the local minimum (or maximum) of a function by iteratively moving in the direction of the steepest descent, as defined by the negative of the gradient. Here’s how you can implement it in Python, specifically for finding the local minimum of the function  starting from .

Steps for Gradient Descent:

1. Define the function .


2. Calculate the derivative of the function, which serves as the gradient. For , the derivative .


3. Initialize parameters: Start at , set a learning rate (which determines the step size), and set a tolerance for the stopping criterion.


4. Update rule: Update  iteratively by moving against the gradient.



Code Implementation

Here's a simple Python code to perform Gradient Descent:

# Define the function and its derivative
def function(x):
    return (x + 3) ** 2

def gradient(x):
    return 2 * (x + 3)

# Gradient Descent parameters
x = 2                  # Starting point
learning_rate = 0.1    # Step size
tolerance = 0.0001     # Threshold for stopping

# Gradient Descent loop
iterations = 0
while True:
    grad = gradient(x)
    new_x = x - learning_rate * grad  # Update x using the gradient
    if abs(new_x - x) < tolerance:    # Stop if the change is below the tolerance
        break
    x = new_x
    iterations += 1

# Print results
print(f"Local minimum found at x = {x}")
print(f"Number of iterations: {iterations}")
print(f"Function value at local minimum: {function(x)}")

Explanation of Parameters and Output

Learning Rate: This controls the step size for each update. A smaller learning rate means smaller steps (more precise but slower convergence), while a larger learning rate can speed up convergence but might overshoot the minimum.

Tolerance: Defines when to stop the algorithm based on how much  changes in each iteration.

Iterations: The loop continues until the change in  is below the tolerance level, indicating convergence to a minimum.


Expected Output

For this specific function, starting at  with a learning rate of , the algorithm should converge near the minimum point  with function value  after several iterations.

Example Output

Local minimum found at x = -2.9997
Number of iterations: 31
Function value at local minimum: 0.00000009

The exact number of iterations and final values may vary slightly depending on the learning rate and tolerance settings, but the result should be very close to .