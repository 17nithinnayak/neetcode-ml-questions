/*Your task is to implement linear regression, a statistical model that ends up being the foundation of neural networks.
  Your must implement get_model_prediction() which returns a prediction value for each dataset value, and get_error() which calculates the error for given prediction data.

Inputs - get_model_prediction:

  X - the dataset to be used by the model to predict the output. len(X) = n, and len(X[i]) = 3 for 0 <= i < n.
weights - the current 
w1, w2 and w3​
  weights for the model. len(weights) = 3.

Inputs - get_error:

model_prediction - the model's prediction for each training example. len(model_prediction) = n.
ground_truth - the correct answer for each example. len(ground_truth) = n.
*/
class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        prediction = np.matmul(X, weights)
        return np.round(prediction, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        error = np.mean(np.square(model_prediction - ground_truth))
        return round(error, 5)
