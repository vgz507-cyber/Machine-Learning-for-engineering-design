import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --- Configuration --- #
INPUT_CSV = 'fowt_master_dataset.csv'
TARGET_NAMES = ['Healthy', 'Anchor Disp', 'Stiffness Deg', 'Broken Mooring', 'Flooded Col', 'Marine Growth']

def run_analysis():
    print("="*75)
    print(" FOWT DECISION TREE: FULL DATA VISUALIZATION")
    print("="*75)

    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_CSV)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        print(f"Loaded {X.shape[1]} predictors. Training tree...")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Train Tree (Max Depth 5, Entropy/Information Gain)
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=5, random_state=42)
    clf.fit(X, y)

    # 3. Generate the Detailed Visual Tree
    plt.figure(figsize=(30, 15))
    
    viz = plot_tree(clf, 
                    feature_names=list(X.columns), 
                    class_names=TARGET_NAMES, 
                    filled=True, 
                    rounded=True, 
                    fontsize=10,
                    impurity=True)

    # 4. Inject Information Gain and Keep Value Array
    tree = clf.tree_
    for i, node in enumerate(viz):
        txt = node.get_text()
        
        # Calculate Info Gain for split nodes
        if '<=' in txt:
            node_entropy = tree.impurity[i]
            num_samples = tree.n_node_samples[i]
            left_id = tree.children_left[i]
            right_id = tree.children_right[i]
            
            w_left = tree.n_node_samples[left_id] / num_samples
            w_right = tree.n_node_samples[right_id] / num_samples
            
            info_gain = node_entropy - (w_left * tree.impurity[left_id] + w_right * tree.impurity[right_id])
            
            # Reconstruct text: Split, Entropy, Samples, Value, Info Gain (No Class)
            lines = txt.split('\n')
            # lines[0] = Split, lines[1] = Entropy, lines[2] = Samples, lines[3] = Value
            # We filter out the 'class =' line if it exists in the split node
            filtered_lines = [l for l in lines if 'class =' not in l]
            new_text = '\n'.join(filtered_lines) + f"\ninfo gain = {info_gain:.3f}"
            node.set_text(new_text)
        else:
            # For Leaf Nodes, we keep everything including the final Class
            pass

    plt.title("FOWT Decision Tree: Complete Engineering Logic & Data Distribution")
    plt.savefig('fowt_decision_tree_final.png', dpi=300, bbox_inches='tight')
    print("SUCCESS: Final visual plot saved as 'fowt_decision_tree_final.png'")
    print("="*75)

if __name__ == "__main__":
    run_analysis()
