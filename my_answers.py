import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch
            X: features batch
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
          
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output,output_error_term)
        
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:,None]
        
        # Weight step (hidden to output)
        delta_weights_h_o +=  output_error_term * hidden_outputs[:,None]
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output +=  self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden +=  self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
# iterations = 2000
# learning_rate = 0.8
# hidden_nodes = 2
# output_nodes = 1

# Progress: 100.0% ... Training loss: 0.144 ... Validation loss: 0.231

# iterations = 3000
# learning_rate = 0.3
# hidden_nodes = 2
# output_nodes = 1

#Progress: 100.0% ... Training loss: 0.221 ... Validation loss: 0.376
            
            
# iterations = 2000
# learning_rate = 0.2
# hidden_nodes = 2
# output_nodes = 1

#progress: 100.0% ... Training loss: 0.262 ... Validation loss: 0.438

# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 2
# output_nodes = 1

#Progress: 100.0% ... Training loss: 0.148 ... Validation loss: 0.270


# iterations = 2000
# learning_rate = 0.9
# hidden_nodes = 2
# output_nodes = 1

#Progress: 100.0% ... Training loss: 0.207 ... Validation loss: 0.391


# iterations = 2000
# learning_rate = 0.7
# hidden_nodes = 2
# output_nodes = 1

#Progress: 100.0% ... Training loss: 0.222 ... Validation loss: 0.381


# iterations = 2000
# learning_rate = 0.8
# hidden_nodes = 2
# output_nodes = 1


# Progress: 100.0% ... Training loss: 0.211 ... Validation loss: 0.373


# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 3
# output_nodes = 1

#Progress: 100.0% ... Training loss: 0.115 ... Validation loss: 0.228
#Progress: 100.0% ... Training loss: 0.105 ... Validation loss: 0.201
#Progress: 100.0% ... Training loss: 0.125 ... Validation loss: 0.279


# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 4
# output_nodes = 1

#Progress: 100.0% ... Training loss: 0.114 ... Validation loss: 0.231


# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 10
# output_nodes = 1


# Progress: 100.0% ... Training loss: 0.145 ... Validation loss: 0.298


# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 15
# output_nodes = 1

#Progress: 100.0% ... Training loss: 0.108 ... Validation loss: 0.233


# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 20
# output_nodes = 1


#Progress: 100.0% ... Training loss: 0.098 ... Validation loss: 0.197


# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 25
# output_nodes = 1


# Progress: 100.0% ... Training loss: 0.079 ... Validation loss: 0.174


# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 25
# output_nodes = 1

# Progress: 100.0% ... Training loss: 0.132 ... Validation loss: 0.269


# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 35
# output_nodes = 1

# Progress: 100.0% ... Training loss: 0.237 ... Validation loss: 0.396


# iterations = 2000
# learning_rate = 0.6
# hidden_nodes = 25
# output_nodes = 1

# Progress: 100.0% ... Training loss: 0.129 ... Validation loss: 0.249


# iterations = 2000
# learning_rate = 0.7
# hidden_nodes = 25
# output_nodes = 1


# Progress: 100.0% ... Training loss: 0.213 ... Validation loss: 0.386

# iterations = 2000
# learning_rate = 0.1
# hidden_nodes = 15
# output_nodes = 1

#Progress: 99.9% ... Training loss: 0.277 ... Validation loss: 0.451



# iterations = 2000
# learning_rate = 0.5
# hidden_nodes = 25
# output_nodes = 1

# Progress: 100.0% ... Training loss: 0.166 ... Validation loss: 0.313
#Progress: 100.0% ... Training loss: 0.104 ... Validation loss: 0.213
#Progress: 100.0% ... Training loss: 0.113 ... Validation loss: 0.228

# Make sure The training loss is below 0.09 and the validation loss is below 0.18.

# iterations = 1500
# learning_rate = 0.9
# hidden_nodes = 15
# output_nodes = 1

# Progress: 99.9% ... Training loss: 0.124 ... Validation loss: 0.199
            
            
            
# iterations = 1500
# learning_rate = 0.9
# hidden_nodes = 12
# output_nodes = 1

# iterations = 3000
# learning_rate = 0.9
# hidden_nodes = 12
# output_nodes = 1

# Progress: 100.0% ... Training loss: 0.064 ... Validation loss: 0.141

iterations = 4000
learning_rate = 0.9
hidden_nodes = 12
output_nodes = 1

#Progress: 100.0% ... Training loss: 0.060 ... Validation loss: 0.141


