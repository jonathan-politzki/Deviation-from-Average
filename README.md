# the goal of this project is to test out a hypothesis found in "https://sarvasvkulpati.com/writer-detection-llm"

# my belief is that each person is different and has different beliefs, interests, and styles, which are reflected in their available data.

# the steps of this project will be to ingest all personal data in unstructured form, starting with twitter and substack, then add a new layer onto a general model (mirosoft phi), then analyze late.

Over a large enough piece of text, a good writer might have many patterns and idiosyncrasies. If there was a way to quantify them, and then average them out, we might have a quantifiable number or vector that represents the style of a person. In this sense, style could be the average deviation from average (with the magnitude and direction captured).

for spotify data I can probably just take my top songs over time.

Model has a predefined set of words (i.e. 50,000,000). This then vectorizes each word. gensim.downloader for vectorizing words. Softmax of ideas vs those of the average. Based on big 5? Constant T for Temperature can make this more spicy. Dot products measure similarity.

Is it valuable in general to know what in your writing is highly deviant from the average?

When you ask the LLM a question (make a query), it quickly scans through all its keys (summaries of information it knows) to find the best matches. Then, it pulls the relevant information (values) from those matches to answer your question. Is there a way to properly key your own data?

** prior work

I generate images of the progression of sentences through feature space. To do so, I:

Loop through every word n the sentence, getting the hidden state vector of 0 to n-1 passed into the model
Perform PCA on the features, and then use the axes with the most variance to plot the points
I used Microsoft Phi-1.5, it seemed like a pretty powerful base model

Here’s some examples of the output:

The reason im actually doing this is because I had to do it manually in December.
