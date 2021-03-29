# movie_review_sentiment
This project closely follows the [NLP lesson](https://course.fast.ai/videos/?lesson=8) in the FastAI course. 

The model uses a pre-trained Wikipedia language model. A language model trains on an entire text corpus then used to predict the next word in a sentence. The model is then fine-tuned on an IMDb review dataset to give it a little more movie lingo.

So the model learned from Wikipedia how to form sentences in English and how to write movie reviews from IMDb. This same model can now be used as a text classifier model to determine whether a review is positive or negative. The notebook for this model can be found in `notebooks/1_mdl_nlp_train_sentiment.ipynb`. Unfortunately, it takes about three hours to train.

Here's the Medium [article](https://medium.com/analytics-vidhya/movie-review-sentiment-analysis-w-rnns-5227e7b52f8c).

### Hosting
I typically add a trained model to a Streamlit showing others how it works, but the text classifier model is almost 500MB, which is too large, so using some cloud hosting will be needed. I added the model to an S3 bucket and included some inference code in `src/src_model_inference.py`. 

The next step of this project is to add the code to some Python instance on AWS and link it to Streamlit. I need to do some reading on AWS first. 
