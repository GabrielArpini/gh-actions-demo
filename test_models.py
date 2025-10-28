import pytest
import numpy as np
# Import functions from your script
from models_comparison import generate_data, fit_svr_model, fit_tree_model

def test_generate_data_shape():
    """Tests if the data generation function returns arrays of the correct shape."""
    X_smooth, y_smooth, _, _ = generate_data('smooth')
    assert X_smooth.shape == (150, 1)
    assert y_smooth.shape == (150,)
    
    X_blocky, y_blocky, _, _ = generate_data('blocky')
    assert X_blocky.shape == (150, 1)
    assert y_blocky.shape == (150,)

def test_model_on_smooth_data():
    """Tests if models can fit a perfect sine wave."""
    print("Testing models on perfect, non-random sine wave...")
    
    # Create a perfect, noiseless sine wave
    X_perfect = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    y_perfect = 3 * np.sin(2 * X_perfect.ravel())
    
    # 1. Test SVR Model
    svr_model, svr_score = fit_svr_model(X_perfect, y_perfect)
    # SVR should be excellent at this.
    assert svr_score > 0.99
    
    # 2. Test Decision Tree Model
    tree_model, tree_score = fit_tree_model(X_perfect, y_perfect, depth=5)
    # The tree will try to approximate a curve with boxes. It won't be as good.
    assert tree_score > 0.8 and tree_score < 0.99

def test_model_on_blocky_data():
    """Tests if models can fit a perfect step function."""
    print("Testing models on perfect, non-random step function...")
    
    n_samples = 100
    X_perfect = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    y_perfect = np.zeros(n_samples)
    y_perfect[30:] = 2.0
    y_perfect[70:] = -1.0
    
    # 1. Test SVR Model
    svr_model, svr_score = fit_svr_model(X_perfect, y_perfect)
    # SVR will try to 'round' the sharp corners, so it won't be perfect.
    assert svr_score > 0.8 and svr_score < 0.96
    
    # 2. Test Decision Tree Model
    tree_model, tree_score = fit_tree_model(X_perfect, y_perfect, depth=3)
    # The tree is PERFECT for this. It should get a score of 1.0
    assert tree_score > 0.999
