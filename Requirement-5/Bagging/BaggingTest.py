from Bagging import Bagging
from sklearn.tree import DecisionTreeClassifier
import unittest
import numpy as np

class TestBagging(unittest.TestCase):
    def setUp(self):
        self.bagging = Bagging(model=DecisionTreeClassifier(), n_estimators=10, max_samples=0.5, random_state=42)

    def test_get_random_subset(self):
        x_data = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 3], [7, 9, 8], [1, 2, 3], [7,9,8]])  
        y_data = np.array([0, 1, 0, 1, 1, 0, 1])
        np.random.seed(42)  
        
        # Get random subset result and assert
        result_x, result_y = self.bagging._get_random_subset(x_data, y_data)
        self.assertNotEqual(x_data.shape[0], result_x.shape[0])                                        # Should have less rows
        self.assertNotEqual(np.unique(x_data, axis=0).shape[0], np.unique(result_x, axis=0).shape[0])  # Should have duplicate rows

    def test_get_soft_vote(self):
        # Create some example predictions
        y_preds = np.array([[[0.2, 0.8], [0.6, 0.4]],  # Prediction from estimator 1
                            [[0.3, 0.7], [0.9, 0.1]],  # Prediction from estimator 2
                            [[0.4, 0.6], [0.2, 0.8]]]) # Prediction from estimator 3        
        expected_result = np.array([1, 0])
        
        # Get the soft vote result and assert
        soft_vote_result = self.bagging._get_soft_vote(y_preds)        
        np.testing.assert_array_equal(soft_vote_result, expected_result)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
