import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import os
from datetime import datetime
import random

# --- Setup ---
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_data(pattern_choice):
    """Generates X, y data based on a chosen pattern."""
    print(f"--- Randomly selected pattern: {pattern_choice.upper()} ---")
    n_samples = 150
    X = np.linspace(-5, 5, n_samples)
    X_reshaped = X.reshape(-1, 1)

    if pattern_choice == 'smooth':
        # Generate SMOOTH sine wave data (favors SVR)
        print("... generating smooth sine wave (favors SVR)")
        true_amplitude = np.random.uniform(2, 5)
        true_frequency = np.random.uniform(0.5, 1.5)
        noise = np.random.normal(0, 0.7, n_samples) # Moderate noise
        y = true_amplitude * np.sin(true_frequency * X) + noise
        true_pattern_name = f"Smooth (Sinusoidal)"

    else: # 'blocky'
        # Generate step function data favoring Decision Tree
        print("... generating blocky step function (favors Decision Tree)")
        y = np.zeros(n_samples)
        # Create 3-4 random steps
        for _ in range(random.randint(3, 4)):
            start_index = random.randint(20, n_samples - 20)
            level = np.random.uniform(-5, 5)
            y[start_index:] = level
        
        noise = np.random.normal(0, 0.5, n_samples) # Less noise
        y += noise
        true_pattern_name = f"Blocky (Step Function)"
    
    print(f"Generated {len(X)} data points.")
    return X_reshaped, y, pattern_choice, true_pattern_name

# teste
def fit_svr_model(X, y):
    """Fits an SVR model and returns its score and predictions."""
    print("Fitting SVR Model")
    svr_model = SVR(kernel='linear', C=10, gamma='auto')
    svr_model.fit(X, y)
    y_pred = svr_model.predict(X)
    score = r2_score(y, y_pred)
    print(f"SVR R-squared: {score:.4f}")
    return svr_model, score

def fit_tree_model(X, y, depth=5):
    """Fits a Decision Tree model and returns its score and predictions."""
    print(f"Fitting Decision Tree w/ depth={depth}...")
    # max_depth=5 gives it flexibility but stops it from overfitting
    tree_model = DecisionTreeRegressor(max_depth=depth)
    tree_model.fit(X, y)
    y_pred = tree_model.predict(X)
    score = r2_score(y, y_pred)
    print(f"Decision Tree R-squared: {score:.4f}")
    return tree_model, score

def create_comparison_plot(X, y, models_and_scores, title_info):
    """Creates and saves a plot comparing all models."""
    print("Generating comparison plot...")
    plt.figure(figsize=(12, 7))
    plt.scatter(X.ravel(), y, alpha=0.5, s=20, label="Generated Data") # .ravel() flattens X

    X_sorted_reshaped = np.sort(X, axis=0)
    
    plot_labels = {'svr': 'SVR Fit', 'tree': 'Decision Tree Fit'}
    plot_colors = {'svr': 'blue', 'tree': 'orange'}
    
    for name, (model, score) in models_and_scores.items():
        y_pred_sorted = model.predict(X_sorted_reshaped)
        plt.plot(X_sorted_reshaped.ravel(), y_pred_sorted, color=plot_colors[name], linewidth=3,
                 label=f'{plot_labels[name]} (R² = {score:.4f})')

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    plt.title(f"Model Comparison (True Pattern: {title_info.upper()})\n(Run at: {run_time})")
    plt.xlabel("X Value")
    plt.ylabel("Y Value")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, "comparison_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved successfully to {plot_path}")
    return plot_path

if __name__ == "__main__":
    pattern = random.choice(['smooth', 'blocky'])
    X_data, y_data, pattern_name, _ = generate_data(pattern)
    
    svr_model, svr_score = fit_svr_model(X_data, y_data)
    tree_model, tree_score = fit_tree_model(X_data, y_data)
    
    models_to_plot = {
        'svr': (svr_model, svr_score),
        'tree': (tree_model, tree_score)
    }
    
    create_comparison_plot(X_data, y_data, models_to_plot, pattern_name)
    print("\nAll tasks completed successfully!")
    
    # End of script
