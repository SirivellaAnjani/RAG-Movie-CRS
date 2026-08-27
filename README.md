# Integrating External Recommenders with RAG for Context-Aware Conversational Movie Recommendation
In this project, I developed a conversational movie recommender that integrates Retrieval-Augmented Generation (RAG) architecture with specialized external recommenders trained on movie datasets. Implemented as the final project in Recommender Systems course in Toronto Metropolitan University.

By combining the reasoning capabilities of LLMs with domain-specific recommenders, I created a conversational recommender that considers context to provide accurate recommendations. I implemented and compared the performance of the system using three external domain recommenders: 1) a BERT-based Transformer with a trainable recommender head, 2) a Relational Graph Convolutional Neural Network (RGCN), and 3) Neural Collaborative Filtering (NCF), all trained on the INSPIRED movie dataset.

I implemented the following models:

* LLM + RAG (Baseline)
* LLM + RAG + RGCN
* LLM + RAG + NCF
* LLM + RAG + Transformer
