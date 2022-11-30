'''
1. Row-wise flatten the data
2. feed the data into the Bi-rnn
3. column-wise flatten the output of the Bi-rnn
4. feed the column-wise flattened output into the Bi-rnn
5. feed the output of the second Bi-rnn into the fully connected layer
6. sigmoid the output of the fully connected layer
7. Reshape the output of the sigmoid layer to the original shape of the input
8. Calculate the loss
9. Backpropagate the loss
10. Update the weights
'''
