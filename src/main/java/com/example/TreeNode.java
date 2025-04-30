package com.example;

import java.io.Serializable;

public class TreeNode implements Serializable {
    // Attributes
    private final String rule;
    private final float prediction;
    private TreeNode leftChild;
    private TreeNode rightChild;

    // Constructors (Nodes are either a decision node, or leaf node)
    public TreeNode(String rule) {
        this.rule = rule;
        this.prediction = 0.0f; 
    }
    public TreeNode(float prediction) {
        this.rule = "";
        this.prediction = prediction; 
    }

    // Getters
    public String getRule() {
        return rule;
    }
    public float getPrediction() {
        return prediction;
    }
    public TreeNode getLeftChild() {
        return leftChild;
    }
    public TreeNode getRightChild() {
        return rightChild;
    }

    // Setters
    public void setLeftChild(TreeNode leftChild) {
        this.leftChild = leftChild;
    }
    public void setRightChild(TreeNode rightChild) {
        this.rightChild = rightChild;
    }
}
