import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, 1024)

        self.bn1 = nn.BatchNorm1d(num_features=1024)

        self.embed = nn.Linear(1024, embed_size)

    def forward(self, images):

        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn1(features)
        features = self.embed(features)

        return features


class DecoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, batch_size, num_layers=2):
        super(DecoderRNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                            dropout=.2, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        self.dropout = nn.Dropout(p=.2)

        self.hidden = self.init_hidden()

    def forward(self, features, captions):

        captions = captions[:, :-1]

        embeds = self.word_embeddings(captions)

        inputs = torch.cat((features.unsqueeze(1), embeds), 1)

        # print('hidden_states Shape ',(self.hidden.shape)) # should be [2,10,512]
        out, self.hidden = self.lstm(inputs)

        out = self.dropout(out)

        out = self.fc(out)

        return out

    def init_hidden(self):

        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def sample(self, inputs, states=None, max_len=20):
        """
                Greedy search:
        Samples captions for pre-processed image tensor (inputs) 
        and returns predicted sentence (list of tensor ids of length max_len)
        """

        predicted_sentence = []

        for i in range(max_len):

            lstm_out, states = self.lstm(inputs, states)

            lstm_out = lstm_out.squeeze(1)
            lstm_out = lstm_out.squeeze(1)
            outputs = self.fc(lstm_out)

            # Get maximum probabilities
            target = outputs.max(1)[1]

            # Append result into predicted_sentence list
            predicted_sentence.append(target.item())

            # Update the input for next iteration
            inputs = self.word_embeddings(target).unsqueeze(1)

        return predicted_sentence
