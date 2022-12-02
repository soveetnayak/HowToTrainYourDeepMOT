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

'''
To train the DHN, we create a data set
with pairs of matrices (D and A
∗
), separated into 114,483
matrices for training and 17,880 for matrices testing. We
generate distance matrices D using ground-truth bounding
boxes and public detections, provided by the MOT challenge datasets [37, 30]. We generate the corresponding assignment matrices A
∗
(as labels for training) using HA described in [6]. We pose the DHN training as a 2D binary
classification task using the focal loss [33]. We compensate
for the class imbalance (between the number of zeros n0
and ones n1 in A
∗
) by weighting the dominant zero-class
using w0 = n1/(n0 + n1). We weight the one-class by
w1 = 1 − w0. We evaluate the performance of DHN by
computing the weighted accuracy (WA)
'''
# import inflection
import torch
import torch.nn as nn

class Deep_Hungarian_net(nn.Module):
    def __init__(self, element_size, hidden_size,targe_size):
        super(Deep_Hungarian_net, self).__init__()
        self.element_size = element_size
        self.hidden_size = hidden_size
        self.bi_rnn_row = nn.GRU(input_size=element_size, hidden_size=hidden_size, num_layers=2, bidirectional=True, dropout=0.5)
        self.bi_rnn_col = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=2, bidirectional=True, dropout=0.5)

        self.fc1 = nn.Linear(hidden_size*2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, targe_size)


        #initialize the weights
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.orthogonal_(m.weight_ih_l0.data)
                torch.nn.init.orthogonal_(m.weight_hh_l0.data)
                torch.nn.init.orthogonal_(m.weight_ih_l0_reverse.data)
                torch.nn.init.orthogonal_(m.weight_hh_l0_reverse.data)

                #initialize gate the bias as -1
                m.bias_ih_l0.data[0:hidden_size] = -1
                m.bias_hh_l0.data[0:hidden_size] = -1
                m.bias_ih_l0_reverse.data[0:hidden_size] = -1
                m.bias_hh_l0_reverse.data[0:hidden_size] = -1

                torch.nn.init.orthogonal_(m.weight_ih_l1.data)
                torch.nn.init.orthogonal_(m.weight_hh_l1.data)
                torch.nn.init.orthogonal_(m.weight_ih_l1_reverse.data)
                torch.nn.init.orthogonal_(m.weight_hh_l1_reverse.data)

                #initialize gate the bias as 1
                m.bias_ih_l1.data[0:hidden_size] = -1
                m.bias_hh_l1.data[0:hidden_size] = -1
                m.bias_ih_l1_reverse.data[0:hidden_size] = -1
                m.bias_hh_l1_reverse.data[0:hidden_size] = -1

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def init_hidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size).cuda()

    def forward(self, x):
        #row-wise flatten the data
        #  shape of x: (batch_size, h,w)
        #  shape of x_row: (h*w, batch_size, 1)
        x_row = x.view(x.size(0), -1, 1)
        x_row = x_row.permute(1,0,2).contiguous()

        #feed the data into the Bi-rnn
        #  shape of x_row: (h*w, batch_size, 1)

        self.hidden_row = self.init_hidden(x_row.size(1))
        x_row_out, self.hidden_row = self.bi_rnn_row(x_row, self.hidden_row)

        #shape of x_row_out: (h*w, batch_size, hidden_size*2)

        # output shape of x_row_out to (h, w, batch_size, hidden_size*2)

        x_row_out = x_row_out.view(x.size(1)*x.size(2) * x_row_out.size(1), x_row_out.size(2))
        x_row_out = x_row_out.view(x.size(1), x.size(2), x.size(0), x_row_out.size(1))

        #column-wise flatten the output of the Bi-rnn
        #  shape of x_col: (h*w, batch_size, hidden_size*2)
        x_col = x_row_out.permute(1,0,2,3).contiguous()
        x_col = x_col.view(-1, x_col.size(2), x_col.size(3))

        #feed the column-wise flattened output into the Bi-rnn
        #  shape of x_col: (w*h, batch_size, hidden_size*2)

        self.hidden_col = self.init_hidden(x_col.size(1))
        x_col_out, self.hidden_col = self.bi_rnn_col(x_col, self.hidden_col)

        #shape of x_col_out: (w*h, batch_size, hidden_size*2)

        # output shape of x_col_out to (h, w, batch_size, hidden_size*2)

        # x_col_out = x_col_out.view(x.size(1)*x.size(2) * x_col_out.size(2), x_col_out.size(3))
        x_col_out = x_col_out.view(x.size(2), x.size(1), x.size(0), -1)
        x_col_out = x_col_out.permute(1,0,2,3).contiguous()

        #shape of x_col_out: (h, w, batch_size, hidden_size*2)\
        x_col_out = x_col_out.view(x_col_out.size(0)*x_col_out.size(1)*x_col_out.size(2), x_col_out.size(3))
        


        lin_out = self.fc1(x_col_out)
        lin_out = self.fc2(lin_out)
        lin_out = self.fc3(lin_out)
        lin_out = lin_out.view(-1, x.size(0))

        sigmoid_out = torch.sigmoid(lin_out)

        return sigmoid_out.view(x.size(1), x.size(2), x.size(0)).permute(2,0,1).contiguous()





