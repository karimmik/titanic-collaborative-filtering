# Collaborative Filtering on Titanic: Latent Factors for Survival Prediction

The Titanic dataset is usually approached with tree-based models (Random Forest, XGBoost, LightGBM) or logistic regression on engineered features.  
In this notebook, I explore a different perspective: **treating passengers as "users" and their attributes (sex, class, ticket group, deck, family, etc.) as "items"**, and then applying **Collaborative Filtering–style techniques**.

### Key idea
Instead of hand-crafting polynomial features, we build a **Passenger × Item interaction matrix**, similar to user–item matrices in recommender systems.  
From this matrix we extract **latent factors** using Truncated SVD (LSA). These factors capture hidden structures, such as:
- family and ticket groups (people traveling together),
- status implied by deck and class,
- gender and age categories.

### Why this is interesting
- **Novel view**: the Titanic problem is reframed as a recommendation problem.  
- **Latent embeddings**: SVD gives compact passenger/item representations that can be visualized (UMAP/t-SNE).  
- **Competitive accuracy**: despite being simple and interpretable, the CF approach reaches ~80% accuracy on the Kaggle test set.

### What you will see
1. Data preprocessing → building passenger–item tokens.  
2. Construction of the sparse interaction matrix.  
3. Collaborative Filtering pipeline: TF-IDF → SVD → Logistic Regression.  
4. Evaluation: cross-validation (~85% accuracy) and Kaggle test submission (~80%).  
5. Visualization of item embeddings to show latent clusters of survival patterns.

---

*This is not about beating Kaggle leaderboards, but about exploring how recommender-system techniques can reveal hidden structure in classical tabular datasets.*