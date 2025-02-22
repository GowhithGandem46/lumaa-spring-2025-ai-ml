# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

**Deadline**: Sunday, Feb 23th 11:59 pm PST

---

## Overview

Build a **content-based recommendation system** that, given a **short text description** of a user’s preferences, suggests **similar items** (e.g., movies) from a small dataset. This challenge should take about **3 hours**, so keep your solution **simple** yet **functional**.

### Example Use Case

- The user inputs:  
  *"I love thrilling action movies set in space, with a comedic twist."*  
- Your system processes this description (query) and compares it to a dataset of items (e.g., movies with their plot summaries or keywords).  
- You then return the **top 3–5 “closest” matches** to the user.

## Solution

### Running the Recommendation System

#### Using Jupyter Notebook

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
2. **Download and Access Files:**
   - `MovieDataset.csv`
   - `Simple_Content_Based_Recommendation.ipynb`
3. **Open the Notebook:**
   - Open `Simple_Content_Based_Recommendation.ipynb` in Jupyter.
4. **Run the Notebook Cells Sequentially:**
   - The notebook walks you through all key steps:
     - **Loading and Preprocessing the Dataset**
     - **Vectorizing the Movie Plots**
     - **Computing Cosine Similarity with the User Query**
     - **Outputting the Top 5 Movie Recommendations**

---

### Example Code Snippet

Below is an excerpt from the notebook that outlines the core implementation:

```python
import numpy as np
import pandas as pd

# Set display options for clarity
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the dataset
file_path = '/content/MovieDataset.csv'  # Update path as needed
df = pd.read_csv(file_path)

# Ensure required columns exist
if 'title' not in df.columns or 'plot' not in df.columns:
    raise ValueError("CSV must contain 'title' and 'plot' columns")

# Fill missing plot values
df['plot'] = df['plot'].fillna('')

# 2. Define the user query
user_query = "I enjoy fights"

# 3. Vectorization: Create TF-IDF vectors for movie plots
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['plot'])

# 4. Transform the user query using the vectorizer
user_query_tfidf = vectorizer.transform([user_query])

# 5. Compute Cosine Similarity between the user query and movie plots
cosine_similarities = cosine_similarity(user_query_tfidf, tfidf_matrix)

# 6. Retrieve the top 5 recommendations
top_n = 5
similarities = cosine_similarities[0]
indices = list(range(len(similarities)))
indices.sort(key=lambda x: similarities[x], reverse=True)
top_indices = indices[:top_n]

# 7. Display the recommendations with similarity scores
for idx in top_indices:
    title = df.iloc[idx]['title']
    similarity_score = similarities[idx]
    print(f"{idx+1}. {title}\n   Similarity: {similarity_score:.4f}")
```

---

### Expected Output

A sample output might appear as follows:

```
1. The Shawshank Redemption
   Similarity: 0.0000
2. The Dark Knight
   Similarity: 0.0000
3. Forrest Gump
   Similarity: 0.0000
...
```

*Note:* The similarity scores depend on the input query and dataset content. Adjust your query or experiment with a richer dataset to achieve more meaningful results.

---

### Demo Video

- A demo video is provided in **DemoRecording.mp4**.
- For an online demo video link, please refer to the **demo.md** file.

---

### Salary Expectation

- Please refer to **Salaryexpectations.ipynb** for details on my monthly salary expectations as part of this submission.

---

### Final Submission

I have ensured that all components run without errors and fully meet the challenge requirements. Thank you for reviewing my submission—I look forward to your feedback!

---
