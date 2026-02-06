# RAG-Chatbot-and-sentiment-analysis

📌 This repository contains two applied AI/NLP projects demonstrating how machine learning and Generative AI can improve knowledge access, issue detection, and operational efficiency in industrial and enterprise software environments.

📁 Project 1: Semiconductor Technical Knowledge Smart-Bot (RAG)
🔍 Overview
This project implements a Retrieval-Augmented Generation (RAG) based Smart-Bot that enables engineers to query semiconductor manufacturing knowledge using natural language.
It simulates how engineering teams can access process, yield, and testing documentation without manually searching large document repositories.

🏗️ Architecture
Document Ingestion – Semiconductor process, yield, and testing PDFs
Text Chunking – Overlapping chunks to preserve technical context
Embeddings – Semantic representation of text
Vector Database (FAISS) – Fast similarity search
RAG Pipeline – Context-grounded answer generation using an LLM

🧠 Why RAG?
Prevents hallucinations by grounding answers in source documents
Enables reuse of historical engineering knowledge
Scales across large volumes of unstructured technical documentation

🚀 Future Enhancements
Inline document citations in responses
Integration with production-grade LLMs
Feedback-driven answer refinement
Integration with operational logs and monitoring systems

📁 Project 2A: Sentiment Analysis of Technical Support Logs
🔍 Overview
This project applies Natural Language Processing (NLP) to classify sentiment in technical support logs and issue-related text.
The goal is to automatically identify negative issues, detect recurring problems, and support prioritization in enterprise software environments.

🏗️ Workflow
Data ingestion (benchmark + support-style text)
Domain-aware text preprocessing
Feature extraction using TF-IDF with n-grams
Sentiment classification using Logistic Regression
Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

🎯 Key Insights
Negative sentiment detection helps prioritize critical issues
NLP converts unstructured logs into actionable signals
Interpretable models provide strong, production-friendly baselines

🚀 Future Enhancements
Aspect-based sentiment analysis
Transformer-based text embeddings
Real-time monitoring dashboards

📁 Project 2B:Sentiment Analysis on Large-Scale Text Data using NLP (IMDb Reviews)

📖 Overview
This project implements a complete Natural Language Processing (NLP) pipeline to classify sentiment in large-scale unstructured text data using the IMDb movie reviews dataset.
The objective is to demonstrate how classical NLP techniques can effectively analyze sentiment at scale using interpretable machine learning models.

🎯 Problem Statement
Unstructured text data contains valuable sentiment signals, but manual analysis is time-consuming and inconsistent.
This project aims to automatically classify text into positive and negative sentiment categories using machine learning.

🏗️ Workflow
Data ingestion and exploration
Text preprocessing and normalization
Feature extraction using TF-IDF
Model training using Logistic Regression
Model evaluation and performance analysis

🔧 Text Preprocessing
Lowercasing
Removal of special characters
Stopword removal
The preprocessing pipeline reduces noise while preserving sentiment-related information.

🧠 Feature Engineering
Used TF-IDF vectorization with unigrams and bigrams
Limited vocabulary size to control sparsity and improve generalization

🤖 Model
Logistic Regression used as a strong, interpretable baseline
Chosen for efficiency and effectiveness with high-dimensional sparse text features

📊 Results
Validation Accuracy: ~89%
Balanced precision and recall for both sentiment classes
Confusion matrix analysis shows no significant class bias

📈 Evaluation Metrics
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Special attention was given to recall to ensure sentiment detection quality.

🚀 Key Learnings
Classical NLP methods remain strong baselines for sentiment analysis
Feature engineering significantly impacts text classification performance
Interpretable models provide valuable insights into prediction behavior

🔮 Future Improvements
Aspect-based sentiment analysis
Transformer-based models (BERT, RoBERTa)
Domain-specific fine-tuning for enterprise feedback data


🎤 “Which project are you most proud of?”
The Smart-Bot highlights my system design and RAG skills, while the sentiment analysis project demonstrates my core NLP and data science fundamentals. Together, they show how AI can improve both knowledge access and issue prioritization in industrial software systems.


