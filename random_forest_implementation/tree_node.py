class TreeNode:
    def __init__(self, rule="", prediction=0.0, left=None, right=None):
        self.rule = rule
        self.prediction = prediction
        self.left = left
        self.right = right