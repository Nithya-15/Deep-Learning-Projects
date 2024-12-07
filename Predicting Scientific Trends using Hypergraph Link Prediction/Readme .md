# Predicting Scientific Research Trends Based on Hyperedge Link Prediction

This project leverages hypergraph-based link prediction to identify emerging trends in scientific research. By building and analyzing hypergraphs, the goal is to forecast relationships between research topics, ultimately aiding in the discovery of future trends in multidisciplinary research areas.

## Project Overview

The methodology of this project is divided into multiple stages, including data cleaning, embedding generation, model building, and evaluation. We apply various models, including HyperGCN, HyperSage, and HNN-LSTM, to predict hyperedge links and analyze trends within the dataset.

## File Structure

The project consists of the following Python notebooks:

1. **`DataCleaning_EDA.ipynb`**  
   - **Purpose**: This notebook performs data cleaning and preprocessing tasks on raw datasets to ensure the quality of input data for embedding generation and model training.
   - **Key Functions**:
     - Removal of noise and inconsistencies from raw data.
     - Extraction of unique keywords and concepts to build the hypergraph.
     - Preprocessing for the generation of the incidence matrix, which represents hypergraph connections.

2. **`Embedding.ipynb`**  
   - **Purpose**: This notebook focuses on generating embeddings for nodes and hyperedges in the hypergraph. It uses embedding techniques like DeepWalk and Word2Vec to capture semantic relationships between keywords and indices.
   - **Key Functions**:
     - Data preprocessing to extract relevant keywords and concepts.
     - Generation of positive and negative samples for embedding training.
     - Use of DeepWalk and Word2Vec for creating low-dimensional embeddings.

3. **`FeedForwardNeuralNetwork.ipynb`**  
   - **Purpose**: This notebook contains the implementation of a feed-forward neural network (FNN) to predict the links in the hypergraph based on the embeddings generated from the previous notebook.
   - **Key Functions**:
     - Design of the neural network architecture with fully connected layers.
     - Training and evaluation of the model using metrics like accuracy, precision, recall, and F1-score.

4. **`HyperGCN.ipynb`**  
   - **Purpose**: This notebook implements the HyperGraph Convolutional Network (HyperGCN) model, which is specifically designed to learn from hypergraph structures using graph convolutions.
   - **Key Functions**:
     - Aggregation of node embeddings using graph convolution layers.
     - Training and evaluation of the model with hypergraph data.
     - Comparative analysis of the HyperGCN model's performance in link prediction tasks.

5. **`HyperSage.ipynb`**  
   - **Purpose**: This notebook implements the HyperSAGE model, which extends the graph attention network framework to hypergraphs, capturing higher-order relationships between nodes and hyperedges.
   - **Key Functions**:
     - Embedding refinement using hyperedge-to-node and node-to-hyperedge message passing.
     - Evaluation of HyperSAGE's ability to predict hyperedge links and capture complex interactions within the dataset.

## How to Run

### Prerequisites
Before running the notebooks, make sure to install the following libraries:

- `numpy==1.26.4`
- `pydantic`
- `PyYAML`
- `torch==2.1.2`
- `torchvision==0.16.2`
- `torchaudio==2.1.2`
- `pandas`
- `dill`
- `tensorflow` or `pytorch` (depending on the framework you are using)
- `sklearn`
- `networkx`
- `matplotlib`
- `gensim` (for Word2Vec)
- `deepwalk` (if using DeepWalk for embeddings)
- `torch-geometric` (if using PyTorch-based models)

You can install these libraries via pip:

```bash
pip install numpy pandas tensorflow sklearn networkx matplotlib gensim deepwalk torch-geometric
