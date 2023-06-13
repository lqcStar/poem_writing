import torch

from data_util import PoetryDataset
from model import RNN

START_TAG = '<START>'
END_TAG = '<EOP>'

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    dataset = PoetryDataset('tang.npz')
    id2word, word2id = dataset.ix2word, dataset.word2ix
    vocab_size = len(word2id)

    model = RNN(vocab_size,
                embedding_dim=256,
                hidden_dim=512,
                num_layers=3,
                bidirectional=False,
                device=device)

    model.load_state_dict(torch.load('params.pth'))
    model = model.to(device)

    input_words = list(input())

    input_ids = torch.tensor([word2id[START_TAG]])
    input_ids = input_ids.long().unsqueeze(0).to(device)
    hidden = None

    with torch.no_grad():
        while True:
            out, hidden = model(input_ids, hidden)
            if input_words:
                next_id = word2id[input_words.pop(0)]
            else:
                next_id = out[-1].argmax().item()
            print(id2word[next_id], end='')
            if next_id == word2id[END_TAG]:
                break

            next_id = torch.tensor([[next_id]]).to(device)
            # input_ids = torch.cat([input_ids, next_id], dim=-1)
            input_ids = next_id

    print()