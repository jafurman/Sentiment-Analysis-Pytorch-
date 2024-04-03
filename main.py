import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input).view(len(input), 1, -1)
        output, hidden = self.rnn(embedded)
        output = self.fc(output.view(len(input), -1))
        return output[-1]  # Return output of the last time step


# Function to convert text to tensor
def text_to_tensor(text, vocab):
    tokens = text.lower().split()
    indexes = [vocab[token] for token in tokens if token in vocab]
    return torch.tensor(indexes, dtype=torch.long)


# Function to predict sentiment
def predict_sentiment(text, model, vocab):
    model.eval()
    with torch.no_grad():
        tensor = text_to_tensor(text, vocab)
        output = model(tensor)
        prediction = torch.sigmoid(output).item()
        sentiment = 'positive' if prediction >= 0.5 else 'negative'
    return sentiment


def defineVocab(data):
    vocabDict = {}
    counter = 0
    for item in data:
        wordList = item.split()
        for word in wordList:
            if word not in vocabDict:
                addWord = str(word).lower()
                vocabDict[addWord] = counter
                counter += 1
    return vocabDict


# Define your own data
data = ["This movie is great", "I didn't like this movie", "Amazing film", "I loved the ambiance", "I did not like the ambiance", "Not to say it wasn't decent, but it was okay"]
# Use continuous sentiment scores: 0 for negative, 1 for positive
labels = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

# Figure out how to get a large CSV file of text optoins... maybe make my own data and train with my own data that I scraped from reddit with my own bias of scores???

# Define your vocabulary
vocab = {}
vocabDict = defineVocab(data)
print(vocabDict)


# Main function
def main():
    # Define model parameters
    INPUT_SIZE = len(vocabDict)
    HIDDEN_SIZE = 128  # Size of the hidden layers in RNN
    OUTPUT_SIZE = 1

    # Initialize model
    model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Iterate over your own data
    for text, label in zip(data, labels):
        # Convert text to tensor
        tensor = text_to_tensor(text, vocabDict)
        # Convert label to tensor
        target = torch.tensor(label, dtype=torch.float32)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(tensor)

        # Calculate loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Print loss
        print(f"Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
