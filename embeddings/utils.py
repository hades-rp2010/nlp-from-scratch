import matplotlib.pyplot as plt
import numpy as np


def reduce_dimensions_pca(vectors):
    """Project high-dimensional vectors down to 2D using SVD / PCA."""
    centered = vectors - np.mean(vectors, axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ Vt[:2].T
    return coords


def reduce_dimensions_tsne(vectors, perplexity=5, random_state=42):
    """Project high-dimensional vectors down to 2D using t-SNE for non-linear cluster separation."""
    try:
        from sklearn.manifold import TSNE

        n_samples = vectors.shape[0]
        perp = min(perplexity, max(1, n_samples - 1))
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        return tsne.fit_transform(vectors)
    except ImportError:
        print("scikit-learn not installed. Falling back to PCA for 2D projection...")
        return reduce_dimensions_pca(vectors)


def find_similar_words(word, embeddings, word2id, id2word, top_k=5):
    """Finds top_k most similar words using cosine similarity."""
    if word not in word2id:
        print(f"Word '{word}' not found in vocabulary.")
        return

    word_idx = word2id[word]
    vec = embeddings[word_idx]

    dot_products = np.dot(embeddings, vec)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec)
    similarities = dot_products / (norms + 1e-8)

    top_indices = np.argsort(similarities)[::-1][1 : top_k + 1]

    print(f"\nWords most similar to '{word}':")
    for idx in top_indices:
        print(f"  {id2word[idx]}: {similarities[idx]:.4f}")


def find_analogy(word_a, word_b, word_c, embeddings, word2id, id2word, top_k=3):
    """Solves vector analogy: word_a is to word_b as word_c is to ? (b - a + c)"""
    for w in [word_a, word_b, word_c]:
        if w not in word2id:
            print(f"Word '{w}' not in vocabulary.")
            return

    vec_a = embeddings[word2id[word_a]]
    vec_b = embeddings[word2id[word_b]]
    vec_c = embeddings[word2id[word_c]]

    target_vec = vec_b - vec_a + vec_c

    dot_products = np.dot(embeddings, target_vec)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec)
    similarities = dot_products / (norms + 1e-8)

    exclude_ids = {word2id[word_a], word2id[word_b], word2id[word_c]}
    sorted_indices = np.argsort(similarities)[::-1]
    results = [idx for idx in sorted_indices if idx not in exclude_ids][:top_k]

    print(f"\nAnalogy: '{word_a}' is to '{word_b}' as '{word_c}' is to:")
    for idx in results:
        print(f"  {id2word[idx]}: {similarities[idx]:.4f}")


def plot_word_embeddings(
    sample_words, embeddings, word2id, id2word, save_path="data/embeddings_plot.png", method="tsne"
):
    """Extracts vectors for sample_words, projects to 2D using t-SNE or PCA."""
    valid_words = [w for w in sample_words if w in word2id]
    if not valid_words:
        print("None of the specified words were found in vocabulary!")
        return

    indices = [word2id[w] for w in valid_words]
    word_vectors = embeddings[indices]

    if method == "tsne":
        coords_2d = reduce_dimensions_tsne(word_vectors)
        title_str = "2D t-SNE Projection of Word Embeddings"
    else:
        coords_2d = reduce_dimensions_pca(word_vectors)
        title_str = "2D PCA Projection of Word Embeddings"

    plt.figure(figsize=(10, 8))
    plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        color="royalblue",
        alpha=0.7,
        edgecolors="k",
        s=100,
    )

    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            xy=(coords_2d[i, 0], coords_2d[i, 1]),
            xytext=(6, 3),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color="darkblue",
        )

    plt.title(title_str, fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Saved word embedding plot to '{save_path}'")
