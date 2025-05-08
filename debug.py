import numpy as np
import pandas as pd

class multinomial_logistic_regression():
    def __init__(
            self, fit_intercept: bool = True,
            iterations = 1000, seed = 33, learning_rate = 1e-3
    ):
        self.fit_intercept = fit_intercept
        self.iterations = iterations
        self.seed = seed
        self.lr = learning_rate

        # Coefficients
        self.B = None
        self.num_classes = None

    # Internal calculations (Private Methods)
    def _add_intercept(self, X):
        """
        Add a column of ones as the first column of the X matrix.
        This guarantees that the last column of X will be a real variable.
        """
        intercept = np.ones((X.shape[0], 1))
        return np.hstack([intercept, X])
    
    def _softmax_probabilities(self, X: np.array):
        """
            p_k(x_n) = exp(x_n · b_k) / Z,

        where 
            Z = 1 + sum_(k=1)^(K-1) exp(x_n · b_k).

        The key is expressing the probabilities as a matrix:
            · x_n is a vector with M elements -> X: N x M matrix.
            · b_k is a vector with M elements -> B: M x K matrix.
            · Z, consequently, is a N x K matrix, and so is P (from p_k).
        
        Consequently:
            P = exp(X · B) / Z,
            
        and, assuming b_K = (0, ..., 0), then:
            
            Z = sum_(k=1)^(K) exp(x_n · b_k). 
        """
        # Calculate X · B product. Can be called logits, as they are
        # not probabilities yet — and are mapped to the real numbers instead.
        logits = np.dot(X, self.B) # shape N x K-1
        
        # Add zeros column (reference class does not update)
        logits = np.hstack([logits, np.zeros((X.shape[0], 1))]) # Shape N x K
        
        # Subtract the maximum logit to avoid overflow. This can be seen
        # as changing the reference variable's coefficients conveniently,
        # so it does not change the result.
        logits -= np.max(logits, axis=1, keepdims=True)

        # Exponentials exp(X · B), also valid for sum in Z
        exp_logits = np.exp(logits) # shape N x K

        # Z function: sum over K (columns)
        # Mathematically: length N, but keepdims=True maintains 
        # shape N x K, so it can be divided term by term 
        # in the next operation
        Z = np.sum(exp_logits, axis=1, keepdims=True) 

        # Return probability
        return exp_logits / Z # shape N x K


    def _compute_gradient(self, X: np.array, y: np.array, 
                          N: int, M: int, K: int):
        """
        grad_cl = sum_(n=1)^(N) x_nl [ 1(y_n=c) - p_c(x_n) ],
        
        where c=1, ..., K-1 and l=1, ..., M.
        
        The key is using matrix operations, so:
            · x_nl -> x_n: M sized vector -> X: N x M matrix.
            · p_c(x_n) -> p(x_n): K-1 sized vector -> P -> N x K-1 matrix.
            · 1(y_n=c) -> delta: N x K-1 onehot matrix.
        Consequently:
            grad = X^T (delta - P) -> M x K-1 matrix.

        Finally, to perform gradient descend toward the 
        negative log-likelihood values, the sign must be the opposite:
            grad = X^T (P - delta) -> M x K-1 matrix
        """

        # P(X) Matrix with shape N x K (not K-1 yet)
        probabilities = self._softmax_probabilities(X)

        # delta Matrix with shape N x K (not K-1 yet)
        delta = np.zeros((N, K)) # initialize matrix
        delta[np.arange(N), y] = 1 # replace with 1 where y_n = c in every row

        # delta - P : restricted to the first K-1 categories -> shape N x K-1
        diff = probabilities[:, :K-1] - delta[:, :K-1]

        # Return gradient: shape M x K-1 (do not update reference class)
        gradient = np.dot(X.T, diff) # shape M x K-1
        # gradient = np.hstack([gradient, np.zeros((M, 1))]) # shape M x K
        return gradient

    # Fit data
    def fit(self, X: np.array, y: np.array):
        """
        Get data (X and y), initialize the coefficients matrix and start training.
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # If N: number of cases, M: number of variables, K: number of classes
        N, M = X.shape
        K = len(np.unique(y))

        # Preallocate memory for the coefficients matrix
        self.B = np.zeros((M, K-1)) # No coeffs for reference class K

        # Start Training
        for i in range(self.iterations):
            gradient = self._compute_gradient(X,y, N,M,K) # shape M x K
            self.B -= self.lr * gradient

    def predicted_probabilities(self, X: np.array):
        if self.fit_intercept:
            X = self._add_intercept(X)
        return self._softmax_probabilities(X)

    def predict(self, X: np.array):
        probs = self.predicted_probabilities(X) # shape N x K
        return np.argmax(probs, axis=1)
    
    # Show results
    def plot(self, X: np.array, y: np.array):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

        # Predicted vs actual
        axs[0].scatter(self.pred, self.y, alpha=0.6)
        axs[0].plot(
            [self.y.min(), self.y.max()],
            [self.y.min(), self.y.max()],
            color='red', linestyle='--',
            label = 'Ideal Fit'
        )

        axs[0].set_title('Predicted vs Actual')
        axs[0].set_xlabel('Predicted values')
        axs[0].set_ylabel('Actual Values')
        axs[0].legend()

        # Plot loss curve
        axs[1].scatter(np.arange(stop=len(self.loss)), self.loss, alpha=0.6, s=10)
        axs[1].set_title('Loss Over Iterations')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Loss')

        # Plot
        plt.tight_layout()
        plt.show()

    def summary(self, X: np.array, y: np.array):
        pred = self.predict(X) # predictions
        K = len(np.unique(y)) # Number of unique categories

        # Confusion matrix
        conf = np.zeros(shape=(K,K), dtype=int)

        for r,p in zip(y,pred):
            conf[r, p] += 1
        
        print("--- Confusion Matrix ---\nreal (rows) \ predicted (columns)")
        print(conf)

        # Number of cases of each REAL category
        sums_real = np.sum(conf, axis=1) # sum by row

        # Number of cases of each PREDICTED category
        sums_pred = np.sum(conf, axis=0) # sum by column

        # Number of cases
        N = np.sum(sums_real)
        
        # Accuracy for all classes (correct cases over number of cases)
        accuracy = np.trace(conf) / N

        # Metrics per class
        precision = np.zeros(shape=(K))
        recall = np.zeros(shape=(K))
        f1score = np.zeros(shape=(K))

        for idx in range(K):
            # True Positives
            TP = conf[idx, idx]

            # Precision = TP / (TP + FP)
            # Recall = TP / (TP + FN)
            # F1Score: armonic mean of precision and recall
            precision[idx] = TP / sums_real[idx] if sums_real[idx] else 0.0
            recall[idx] = TP / sums_pred[idx] if sums_pred[idx] else 0.0
            f1score[idx] = 2 * precision[idx] * recall[idx] / (precision[idx] + recall[idx]) if (precision[idx] + recall[idx]) else 0.0
        
        # Show metrics
        print("--- Performance by class ---")
        print(f'Class:\t\t {[k for k in range(K)]}')
        print(f'Precision:\t {precision}')
        print(f'Recall:\t\t {recall}')
        print(f'F1-Score:\t {f1score}')

        print("--- Global Model Performance (Average) ---")
        print(f'Accuracy:\t {accuracy}')
        print(f'Precision:\t {np.mean(precision)}')
        print(f'Recall:\t\t {np.mean(recall)}')
        print(f'F1-Score:\t {np.mean(f1score)}')

# --- Run Training ---
# Example Data
N = 5000
np.random.seed(33)
y = np.random.randint(low=0, high=5, size=N) # <-- Multiple Categories (4)
X = pd.DataFrame.from_dict({'x1': y * 2, 'x2': np.random.rand(N)})
X = X.values

# Build model
multinomial = multinomial_logistic_regression(iterations=1000)
multinomial.fit(X = X, y = y)
pred = multinomial.predict(X)

# Performance
multinomial.summary(X, y)
