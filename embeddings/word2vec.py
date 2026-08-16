import os

import numpy as np
import torch
from dataset import (
    UNK_TOKEN,
    build_vocab,
    download_data,
    generate_cbow_pairs,
    generate_skip_gram_pairs,
    load_data,
    subsample_tokens,
)
from torch import nn, optim
from tqdm import tqdm
from utils import find_analogy, find_similar_words, plot_word_embeddings

# Hyperparams for Fast Local Mac Training
MODE = "skipgram_neg"  # Options: "cbow", "skipgram", or "skipgram_neg"
NUM_EPOCHS = 20
LEARNING_RATE = 3e-3
BATCH_SIZE = 512
EMBED_DIM = 32
MIN_FREQ = 5
WINDOW_SIZE = 4

# Path to save models to
SAVE_MODEL_PATH = "./models"


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using PyTorch device: {device}")


class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.context_embed = nn.Embedding(vocab_size, embed_dim)
        self.target_dense = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, context_ids):
        embeds = self.context_embed(context_ids)  # (batch_size, 2 * window_size, embed_dim)
        embed_avg = embeds.mean(dim=1)  # (batch_size, embed_dim)
        logits = self.target_dense(embed_avg)  # (batch_size, vocab_size)
        return logits

    def get_embeddings(self):
        return self.context_embed.weight.detach().cpu().numpy()


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.target_embed = nn.Embedding(vocab_size, embed_dim)
        self.target_dense = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, target_ids):
        embed = self.target_embed(target_ids)  # (batch_size, embed_dim)
        logits = self.target_dense(embed)  # (batch_size, vocab_size)
        return logits

    def get_embeddings(self):
        return self.target_embed.weight.detach().cpu().numpy()


class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.target_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(
        self,
        target_ids,  # (batch_size,)
        pos_context_ids,  # (batch_size,)
        neg_context_ids,
    ):  # (batch_size, K)

        target_embed = self.target_embed(target_ids)  # (batch_size, embed_dim)
        pos_context_embed = self.context_embed(pos_context_ids)  # (batch_size, embed_dim)
        neg_context_embed = self.context_embed(neg_context_ids)  # (batch_size, K, embed_dim)

        pos_logits = torch.sum(target_embed * pos_context_embed, dim=1)  # (batch_size,)
        neg_logits = torch.bmm(neg_context_embed, target_embed.unsqueeze(2)).squeeze(
            2
        )  # (batch_size, K)
        return pos_logits, neg_logits

    def get_embeddings(self):
        return self.target_embed.weight.detach().cpu().numpy()


def push_to_gpu(*tensors):
    if len(tensors) == 1:
        return tensors[0].to(device)
    return tuple(t.to(device) for t in tensors)


def get_batch_slice(start_idx, end_idx, *tensors):
    if len(tensors) == 1:
        return tensors[0][start_idx:end_idx]
    return tuple(t[start_idx:end_idx] for t in tensors)


def save_model(model, filename="model.pt"):
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    filepath = os.path.join(SAVE_MODEL_PATH, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Saved model state dictionary to '{filepath}'")


def train_cbow(model, contexts_tensor, targets_tensor, num_epochs=5, batch_size=1024, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    contexts_tensor, targets_tensor = push_to_gpu(contexts_tensor, targets_tensor)

    num_samples = len(contexts_tensor)
    num_batches = (num_samples + batch_size - 1) // batch_size

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i in pbar:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            context_b, target_b = get_batch_slice(
                start_idx, end_idx, contexts_tensor, targets_tensor
            )

            optimizer.zero_grad()
            logits = model(context_b)
            loss = criterion(logits, target_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} Complete | Average Loss: {avg_loss:.4f}")

    return model


def train_skipgram(model, input_tensors, output_tensors, num_epochs=5, batch_size=512, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    input_tensors, output_tensors = push_to_gpu(input_tensors, output_tensors)

    num_samples = len(input_tensors)
    num_batches = (num_samples + batch_size - 1) // batch_size

    model.train()
    for epoch in range(num_epochs):
        total_loss = torch.tensor(0.0, device=device)
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            input_batch, output_batch = get_batch_slice(
                start_idx, end_idx, input_tensors, output_tensors
            )

            optimizer.zero_grad()
            preds = model(input_batch)
            loss = criterion(preds, output_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss

        avg_loss = total_loss.item() / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} Complete | Average Loss: {avg_loss:.4f}")

    return model


def train_skigram_negsampling(
    model,
    target_tensors,
    pos_context_tensors,
    neg_context_tensors,
    num_epochs=5,
    batch_size=1024,
    lr=1e-3,
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    target_tensors, pos_context_tensors, neg_context_tensors = push_to_gpu(
        target_tensors, pos_context_tensors, neg_context_tensors
    )

    num_samples = len(target_tensors)
    num_batches = (num_samples + batch_size - 1) // batch_size

    model.train()
    for epoch in range(num_epochs):
        total_loss = torch.tensor(0.0, device=device)
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            target_batch, pos_context_batch, neg_context_batch = get_batch_slice(
                start_idx,
                end_idx,
                target_tensors,
                pos_context_tensors,
                neg_context_tensors,
            )

            optimizer.zero_grad()
            pos_logits, neg_logits = model(target_batch, pos_context_batch, neg_context_batch)
            loss = criterion(pos_logits, torch.ones_like(pos_logits)) + criterion(
                neg_logits, torch.zeros_like(neg_logits)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss

        avg_loss = total_loss.item() / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} Complete | Average Loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    dataset_name = "wikitext-2"
    download_data(dataset_name)
    raw_dataset = load_data(dataset_name)

    # Build vocab on dataset
    word2id, id2word, words = build_vocab(raw_dataset, min_freq=5)
    words = subsample_tokens(words)

    token_ids = [word2id.get(w, word2id[UNK_TOKEN]) for w in words]

    if MODE == "cbow":
        pairs = generate_cbow_pairs(token_ids, window_size=WINDOW_SIZE)
        contexts_tensor = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        targets_tensor = torch.tensor([p[1] for p in pairs], dtype=torch.long)

        cbow_model = CBOW(vocab_size=len(word2id), embed_dim=EMBED_DIM)
        trained_model = train_cbow(
            cbow_model,
            contexts_tensor,
            targets_tensor,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        save_model(trained_model, "cbow.pt")
        embeddings = trained_model.get_embeddings()

    elif MODE == "skipgram":
        pairs = generate_skip_gram_pairs(token_ids, window_size=WINDOW_SIZE)
        target_tensor = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        context_tensor = torch.tensor([p[1] for p in pairs], dtype=torch.long)

        skipgram_model = SkipGram(vocab_size=len(word2id), embed_dim=EMBED_DIM)
        trained_model = train_skipgram(
            skipgram_model,
            target_tensor,
            context_tensor,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        save_model(trained_model, "skipgram.pt")
        embeddings = trained_model.get_embeddings()

    elif MODE == "skipgram_neg":
        from dataset import build_noise_table

        pairs = generate_skip_gram_pairs(token_ids, window_size=WINDOW_SIZE)
        target_tensor = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        pos_context_tensor = torch.tensor([p[1] for p in pairs], dtype=torch.long)

        K = 5
        print(f"Building 3/4 power noise table and sampling {K} negative noise words per pair...")
        noise_table = build_noise_table(words, word2id)
        neg_context_tensor = torch.tensor(
            np.random.choice(noise_table, size=(len(pairs), K)), dtype=torch.long
        )

        skipgram_neg_model = SkipGramNegSampling(vocab_size=len(word2id), embed_dim=EMBED_DIM)
        trained_model = train_skigram_negsampling(
            skipgram_neg_model,
            target_tensor,
            pos_context_tensor,
            neg_context_tensor,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )
        save_model(trained_model, "skipgram_neg.pt")
        embeddings = trained_model.get_embeddings()

    # Test modern word similarities (from Mikolov et al. paper benchmarks)!
    find_similar_words("france", embeddings, word2id, id2word)
    find_similar_words("computer", embeddings, word2id, id2word)
    find_similar_words("king", embeddings, word2id, id2word)

    # Test classic Paper Analogies! (b - a + c)
    find_analogy("man", "king", "woman", embeddings, word2id, id2word)  # -> queen
    find_analogy("france", "paris", "japan", embeddings, word2id, id2word)  # -> tokyo
    find_analogy("small", "smaller", "large", embeddings, word2id, id2word)  # -> larger

    # Visualize modern semantic categories in 2D
    sample_words = [
        # Countries & Capitals (Paper Benchmark)
        "france",
        "paris",
        "italy",
        "rome",
        "japan",
        "tokyo",
        "england",
        "london",
        # Family & Gender (Paper Benchmark)
        "man",
        "woman",
        "king",
        "queen",
        "boy",
        "girl",
        "father",
        "mother",
        # Technology & Science
        "computer",
        "software",
        "science",
        "technology",
        "music",
        "art",
    ]
    plot_word_embeddings(
        sample_words,
        embeddings,
        word2id,
        id2word,
        save_path=f"data/embeddings_plot_{MODE}.png",
    )
