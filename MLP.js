class TDataPoint {
    constructor(input, target) {
        this.input = input;
        this.target = target;
    }
}

class TNeuron {
    constructor() {
        this.weights = [];
        this.bias = 0;
        this.output = 0;
        this.error = 0;
    }
}

class TLayer {
    constructor() {
        this.neurons = [];
    }
}

class TMultiLayerPerceptron {
    constructor(inputSize, hiddenSizes, outputSize) {
        this.learningRate = 0.1;
        this.maxIterations = 30;
        this.inputLayer = new TLayer();
        this.hiddenLayers = [];
        this.outputLayer = new TLayer();
        this.initialize(inputSize, hiddenSizes, outputSize);
    }

    initialize(inputSize, hiddenSizes, outputSize) {
        hiddenSizes.forEach((size) => {
            this.hiddenLayers.push(new TLayer());
            this.hiddenLayers[this.hiddenLayers.length - 1].neurons = [];
            for (let i = 0; i < size; i++) {
                this.hiddenLayers[this.hiddenLayers.length - 1].neurons.push(new TNeuron());
                this.hiddenLayers[this.hiddenLayers.length - 1].neurons[i].weights = new Array(inputSize).fill(0);
                this.hiddenLayers[this.hiddenLayers.length - 1].neurons[i].bias = 0;
            }
        });
        this.outputLayer.neurons = [];
        for (let i = 0; i < outputSize; i++) {
            this.outputLayer.neurons.push(new TNeuron());
            this.outputLayer.neurons[i].weights = new Array(this.hiddenLayers[this.hiddenLayers.length - 1].neurons.length).fill(0);
            this.outputLayer.neurons[i].bias = 0;
        }
    }

    feedForward(input) {
        let sum;
        for (let k = 0; k < this.hiddenLayers.length; k++) {
            for (let i = 0; i < this.hiddenLayers[k].neurons.length; i++) {
                sum = this.hiddenLayers[k].neurons[i].bias;
                if (k === 0) {
                    for (let j = 0; j < input.length; j++) {
                        sum += input[j] * this.hiddenLayers[k].neurons[i].weights[j];
                    }
                } else {
                    for (let j = 0; j < this.hiddenLayers[k - 1].neurons.length; j++) {
                        sum += this.hiddenLayers[k - 1].neurons[j].output * this.hiddenLayers[k].neurons[i].weights[j];
                    }
                }
                this.hiddenLayers[k].neurons[i].output = 1 / (1 + Math.exp(-sum));
            }
        }
        // Calculate output layer outputs
        for (let i = 0; i < this.outputLayer.neurons.length; i++) {
            sum = this.outputLayer.neurons[i].bias;
            for (let j = 0; j < this.hiddenLayers[this.hiddenLayers.length - 1].neurons.length; j++) {
                sum += this.hiddenLayers[this.hiddenLayers.length - 1].neurons[j].output * this.outputLayer.neurons[i].weights[j];
            }
            this.outputLayer.neurons[i].output = 1 / (1 + Math.exp(-sum));
        }
    }

    backPropagate(target) {
        // Calculate output layer errors
        for (let i = 0; i < this.outputLayer.neurons.length; i++) {
            this.outputLayer.neurons[i].error = this.outputLayer.neurons[i].output * (1 - this.outputLayer.neurons[i].output) * (target[i] - this.outputLayer.neurons[i].output);
        }
        // Calculate hidden layer errors
        for (let k = this.hiddenLayers.length - 1; k >= 0; k--) {
            for (let i = 0; i < this.hiddenLayers[k].neurons.length; i++) {
                if (k === this.hiddenLayers.length - 1) {
                    let errorSum = 0;
                    for (let j = 0; j < this.outputLayer.neurons.length; j++) {
                        errorSum += this.outputLayer.neurons[j].error * this.outputLayer.neurons[j].weights[i];
                    }
                    this.hiddenLayers[k].neurons[i].error = this.hiddenLayers[k].neurons[i].output * (1 - this.hiddenLayers[k].neurons[i].output) * errorSum;
                } else {
                    let errorSum = 0;
                    for (let j = 0; j < this.hiddenLayers[k + 1].neurons.length; j++) {
                        errorSum += this.hiddenLayers[k + 1].neurons[j].error * this.hiddenLayers[k + 1].neurons[j].weights[i];
                    }
                    this.hiddenLayers[k].neurons[i].error = this.hiddenLayers[k].neurons[i].output * (1 - this.hiddenLayers[k].neurons[i].output) * errorSum;
                }
            }
        }
    }

    updateWeights() {
        // Update weights of output layer neurons
        for (let i = 0; i < this.outputLayer.neurons.length; i++) {
            for (let j = 0; j < this.hiddenLayers[this.hiddenLayers.length - 1].neurons.length; j++) {
                this.outputLayer.neurons[i].weights[j] += this.learningRate * this.outputLayer.neurons[i].error * this.hiddenLayers[this.hiddenLayers.length - 1].neurons[j].output;
            }
            this.outputLayer.neurons[i].bias += this.learningRate * this.outputLayer.neurons[i].error;
        }
        // Update weights of hidden layer neurons
        for (let k = this.hiddenLayers.length - 1; k >= 0; k--) {
            for (let i = 0; i < this.hiddenLayers[k].neurons.length; i++) {
                if (k === 0) {
                    for (let j = 0; j < this.inputLayer.neurons.length; j++) {
                        this.hiddenLayers[k].neurons[i].weights[j] += this.learningRate * this.hiddenLayers[k].neurons[i].error * this.inputLayer.neurons[j].output;
                    }
                    this.hiddenLayers[k].neurons[i].bias += this.learningRate * this.hiddenLayers[k].neurons[i].error;
                } else {
                    for (let j = 0; j < this.hiddenLayers[k - 1].neurons.length; j++) {
                        this.hiddenLayers[k].neurons[i].weights[j] += this.learningRate * this.hiddenLayers[k].neurons[i].error * this.hiddenLayers[k - 1].neurons[j].output;
                    }
                    this.hiddenLayers[k].neurons[i].bias += this.learningRate * this.hiddenLayers[k].neurons[i].error;
                }
            }
        }
    }

    predict(input) {
        this.inputLayer.neurons = [];
        for (let i = 0; i < input.length; i++) {
            this.inputLayer.neurons.push(new TNeuron());
            this.inputLayer.neurons[i].output = input[i];
        }
        this.feedForward();
        let output = [];
        for (let i = 0; i < this.outputLayer.neurons.length; i++) {
            output.push(this.outputLayer.neurons[i].output);
        }
        return output;
    }

    train(input, target) {
        this.feedForward();
        this.backPropagate(target);
        this.updateWeights();
    }

    saveMLPModel(filename) {
        let file = fs.createWriteStream(filename);
        file.write(`${this.hiddenLayers.length}\\n`);
        file.write(`${this.inputLayer.neurons.length}\\n`);
        for (let i = 0; i < this.hiddenLayers.length; i++) {
            file.write(`${this.hiddenLayers[i].neurons.length}\\n`);
            for (let j = 0; j < this.hiddenLayers[i].neurons.length; j++) {
                file.write(`${this.hiddenLayers[i].neurons[j].weights.length}\\n`);
                for (let k = 0; k < this.hiddenLayers[i].neurons[j].weights.length; k++) {
                    file.write(`${this.hiddenLayers[i].neurons[j].weights[k]}\\n`);
                }
                file.write(`${this.hiddenLayers[i].neurons[j].bias}\\n`);
            }
        }
        file.write(`${this.outputLayer.neurons.length}\\n`);
        for (let i = 0; i < this.outputLayer.neurons.length; i++) {
            file.write(`${this.outputLayer.neurons[i].weights.length}\\n`);
            for (let j = 0; j < this.outputLayer.neurons[i].weights.length; j++) {
                file.write(`${this.outputLayer.neurons[i].weights[j]}\\n`);
            }
            file.write(`${this.outputLayer.neurons[i].bias}\\n`);
        }
        file.end();
    }

    static loadMLPModel(filename) {
        let file = fs.createReadStream(filename);
        let data = file.read();
        let hiddenLayers = [];
        let inputSize = parseInt(data.split('\\n')[1]);
        let hiddenSizes = [];
        for (let i = 2; i < data.length; i++) {
            hiddenSizes.push(parseInt(data.split('\\n')[i]));
        }
        let outputSize = parseInt(data.split('\\n')[data.length - 1]);
        let mlp = new TMultiLayerPerceptron(inputSize, hiddenSizes, outputSize);
        for (let i = 0; i < mlp.hiddenLayers.length; i++) {
            mlp.hiddenLayers[i].neurons = [];
            for (let j = 0; j < hiddenSizes[i]; j++) {
                mlp.hiddenLayers[i].neurons.push(new TNeuron());
                mlp.hiddenLayers[i].neurons[j].weights = [];
                for (let k = 0; k < inputSize; k++) {
                    mlp.hiddenLayers[i].neurons[j].weights.push(parseFloat(data.split('\\n')[i * (inputSize + 2) + 2 + k]));
                }
                mlp.hiddenLayers[i].neurons[j].bias = parseFloat(data.split('\\n')[i * (inputSize + 2) + (inputSize + 2)]);
            }
        }
        for (let i = 0; i < mlp.outputLayer.neurons.length; i++) {
            mlp.outputLayer.neurons[i].weights = [];
            for (let j = 0; j < hiddenSizes[hiddenSizes.length - 1]; j++) {
                mlp.outputLayer.neurons[i].weights.push(parseFloat(data.split('\\n')[mlp.hiddenLayers.length * (inputSize + 2) + (inputSize + 2) + i * (hiddenSizes[hiddenSizes.length - 1) + 1] + j]));
            }
            mlp.outputLayer.neurons[i].bias = parseFloat(data.split('\\n')[mlp.hiddenLayers.length * (inputSize + 2) + (inputSize + 2) + i * (hiddenSizes[hiddenSizes.length - 1) + 1] + hiddenSizes[hiddenSizes.length - 1)]);
        }
        return mlp;
    }
}

// Example usage
let data = [];
for (let i = 0; i < 7500; i++) {
    data.push(new TDataPoint([Math.random() * 0.5, Math.random() * 0.5, Math.random() * 0.5, Math.random() * 0.5], [1, 0, 0]));
}
let mlp = new TMultiLayerPerceptron(4, [8, 8, 8], 3);
mlp.maxIterations = 30;
for (let i = 0; i < 30; i++) {
    for (let j = 0; j < data.length; j++) {
        mlp.train(data[j].input, data[j].target);
    }
}
let output = mlp.predict([0.5, 0.5, 0.5, 0.5]);
console.log(output);
