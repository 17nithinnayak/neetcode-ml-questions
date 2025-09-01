/*Your task is to minimize the function via Gradient Descent: 
f(x)=x2

Gradient Descent is an optimization technique widely used in machine learning for training models. It is crucial for minimizing the cost or loss function and finding the optimal parameters of a model.

For the above function the minimizer is clearly x = 0, but you must implement an iterative approximation algorithm, through gradient descent.

Input:

iterations - the number of iterations to perform gradient descent. iterations >= 0.
learning_rate - the learning rate for gradient descent. 1 > learning_rate > 0.
init - the initial guess for the minimizer. init != 0.
Given the number of iterations to perform gradient descent, the learning rate, and an initial guess, return the value of x that globally minimizes this function.

Round your final result to 5 decimal places using Python's round() function.
Solution Approach: The first step in gradient descent is to calculate the derivative (gradient) of the function. The derivative gives the slope of the function.
                   Start with an initial guess. In this problem the initial guess is given as a parameter.
                   At each step, the current value (or guess) is updated by subtracting the product of the derivative and the learning rate.
                   This process is repeated (iterated) a variable number of times. Upon each iteration we will move closer to the point where the derivative is zero, which is the minimum of the function.*/

class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        minimizer = init

        for i in range(iterations):
            derivative = 2 * minimizer
            minimizer = minimizer - learning_rate * derivative

        return round(minimizer, 5) 
