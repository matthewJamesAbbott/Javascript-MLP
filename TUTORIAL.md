# Multi-Layer Perceptron (MLP) Tutorial & Capabilities Guide

## What This Application Can Do

Welcome to the **Javascript-MLP** application! This is a comprehensive Multi-Layer Perceptron trainer with a web-based interface that provides powerful machine learning capabilities. Here's everything this application can do for you:

## 🧠 Complete Neural Network Capabilities

### 1. **Network Architecture Configuration**
- **Flexible Input/Output Sizes**: Configure any number of input features and output classes
- **Multi-Layer Hidden Networks**: Create deep networks with multiple hidden layers (e.g., "8,8,8" for three hidden layers)
- **Multiple Activation Functions**: 
  - Hidden layers: Sigmoid, Tanh, ReLU
  - Output layer: Softmax, Sigmoid, Tanh, ReLU
- **Adjustable Learning Rate**: Fine-tune training speed (0.001 to 1.0)

### 2. **Advanced Training Features**
- **Batch Training**: Support for different batch sizes:
  - Online learning (batch size 1)
  - Mini-batch (32, 64, 128)
  - Full batch processing
- **Progress Tracking**: Real-time training progress with epoch counting
- **Configurable Epochs**: Set different epoch counts for training and validation

### 3. **Data Management**
- **CSV File Loading**: Import training data from CSV files
- **Direct Data Input**: Paste CSV data directly into the interface
- **Automatic Test Data Generation**: Generate synthetic datasets for experimentation
- **Flexible Data Format**: Automatically handles input/output size matching

### 4. **Comprehensive Model Evaluation**
- **K-Fold Cross Validation**: Robust model validation with configurable fold counts
- **Detailed Metrics**: 
  - Precision per class
  - Recall per class  
  - F1-Score per class
  - Overall accuracy
- **Performance Tables**: Clear visualization of all evaluation metrics

### 5. **Model Persistence**
- **Save Models**: Export trained models as JSON files
- **Load Models**: Import previously saved models for continued use
- **Complete State Preservation**: All weights, biases, and architecture saved

### 6. **Real-time Prediction**
- **Interactive Prediction**: Test models with custom input data
- **Confidence Scores**: Get prediction confidence percentages
- **Raw Output Display**: View raw neural network outputs before classification

## 📋 Step-by-Step Usage Guide

### Getting Started
1. **Open the Application**: Load `MLP.html` in your web browser
2. **Configure Network**: Set input size, hidden layers, output size, and activation functions
3. **Create Network**: Click "Create Network" to initialize your neural network

### Training Your Model
1. **Load Data**: Either upload a CSV file or generate test data
2. **Configure Training**: Set epochs, batch size, and validation parameters
3. **Train**: Use "Train with Progress" to see real-time training updates

### Evaluating Performance
1. **Run Metrics**: Click "Run All Evaluation Metrics" for comprehensive analysis
2. **Cross Validation**: Use "Run K-Fold Cross Validation" for robust validation
3. **View Results**: Analyze precision, recall, F1-scores, and accuracy

### Making Predictions
1. **Input Data**: Enter comma-separated values matching your input size
2. **Predict**: Click "Predict" to get classification results
3. **Interpret**: View both raw outputs and predicted class with confidence

### Saving Your Work
1. **Save Model**: Export your trained model for later use
2. **Load Model**: Import previously saved models to continue working

## 🎯 Example Use Cases

### Educational Applications
- **Learning Neural Networks**: Understand how MLPs work with interactive visualization
- **Experimenting with Architectures**: Try different layer configurations and activations
- **Understanding Training**: Watch the training process with progress tracking

### Research & Development
- **Prototype Testing**: Quickly test neural network ideas
- **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and architectures
- **Performance Analysis**: Comprehensive evaluation with multiple metrics

### Data Science Projects
- **Classification Tasks**: Handle multi-class classification problems
- **Model Validation**: Use K-fold cross-validation for robust performance assessment
- **Quick Prototyping**: Rapid development and testing of neural network solutions

## 🔧 Technical Features

### Architecture Support
- **Deep Networks**: Multiple hidden layers with independent activation functions
- **Flexible Sizing**: Any combination of input/hidden/output layer sizes
- **Modern Activations**: ReLU, Sigmoid, Tanh, and Softmax functions

### Training Algorithms
- **Backpropagation**: Standard gradient descent with error backpropagation
- **Batch Processing**: Efficient training with configurable batch sizes
- **Weight Updates**: Proper gradient descent weight and bias updates

### Evaluation Methods
- **Cross Validation**: K-fold validation for unbiased performance estimates
- **Multiple Metrics**: Precision, recall, F1-score for comprehensive evaluation
- **Per-Class Analysis**: Detailed performance breakdown by classification class

## 💡 Tips for Best Results

1. **Start Simple**: Begin with smaller networks and gradually increase complexity
2. **Balance Data**: Ensure your training data is well-balanced across classes
3. **Tune Learning Rate**: Start with 0.1 and adjust based on training behavior
4. **Use Cross Validation**: Always validate your model with K-fold cross-validation
5. **Save Frequently**: Save your models after successful training sessions

## 🚀 Advanced Features

- **Real-time Training Progress**: Watch your model learn with live updates
- **Comprehensive Logging**: Track training completion and data statistics
- **Interactive Interface**: User-friendly web interface requiring no installation
- **JSON Model Export**: Standard format for model sharing and archival

This application provides everything you need for neural network experimentation, learning, and practical application development. Whether you're a student learning about neural networks or a researcher prototyping new ideas, this tool offers comprehensive capabilities in an accessible, web-based interface.