---
layout: post
title: Why Use GANs for Time Series?! Also How?
categories: [Deep Learning]
tags: [Deep Learning]
---
>I know I'm bad at remembering and articulating stuff I learned so I decided to write stuff here, if I suck at grammar or suffer from lack of vocabulary variety... DEAL WITH IT, it's my blog!

##   INTRO

So why use GANs for Time Series when there are plenty of good alternatives like RNNs, LSTMs and AutoRegression Models? The loop-like structure of these makes them perfect for sequential data, doesn't it?

##   NOPE!

The first argument about Time Series forecasting with these models is that they are fundamentally deterministic and more like regression (indication) than generation (prediction). New samples can not be randomly samples from them without external conditioning. In AutoRegression Models, we keep predicting the next timestamp as a function of previous timestamps; therefore, we only care about temporary dynamics of our dataset when we're iterating over each cell of the sequence (even LSTMs only care as far as their long-term memory or Cell state allows them).

> "A good generative model for time-series data should preserve **temporal dynamics** in the sense that new sequences respect the original relationships between variables across time." - The guy from the paper I read

The second argument against them is that they introduce no randomness to prediction - they are not a one-to-many solution; meaning that for a point in a time-series, there are not many options to predict but the one your model indicated from the past. The fact that Time Series data have uncertainty as their nature (unlike images or natural language) should also be embedded in the models that are analyzing them, and therefore why some people suggest GANs.

##   LET THE MAN COOK!

GANs aren't that perfect either; they are inherently unstable models. They have problems like non-convergence and vanishing gradients (meaning Discriminator is way more successful than Generator so G keeps losing and thus learns nothing), and mode collapses (when G learns to fool D with only one class of the data and stops learning more and keeps sending that variant repeatedly to the D) which are not completely solved yet. Also Contrary to image-based GANs, there is no unified metric for TS GANs (LMAO! image-based GANs evaluation metric is human's eye), and that makes their performance evaluation subjective. GANs for TS has been mentioned by a lot of papers in past few years, but now it's getting more attention and becoming a growing body of research... I promise!

##   VARIANTS OF TIME SERIES BASED GANS

Time Series GANs come in two variants: Discrete and Continuous. GANs struggle with discrete data because there are zero gradients almost everywhere due to the nature of the data (it's discrete how am i supposed to take DERIVATIES!!!), so you can't use backpropagation alone. Here I list some interesting TS-GAN models which performed competitively: 

#### Discrete Variants:

- SeqGAN: This one was made to fix the discrete backprop problem. Here G is a LSTM, D is a CNN, and G's gradient gets updated by a policy gradient and Monte Carlo Search 
- QuantGAN: This one came along to capture those long-term dependencies (long long long-term memories) which LSTMs couldn't preserve. Both G & D are temporal convolutional networks with skip connections. Authors claim they can outperform other conventional mathematical finance models.

#### Continuous Variants:
- RNN-GAN: The generator is an RNN and the Discriminator is a bi-directional RNN which allows D to interpret G's generated sequence in both directions. The RNNs here were two stacked LSTMs with 350 hidden layers.
- C-RNN-GAN: This one is the previous one but with backpropagation through time(wtf?). They applied a lot of optimization techniques to prevent one model from getting stronger than the other, and then trained it on a bunch of midi files from 160 composers to generate music. Unlike their math in the paper their music doesn't suck...
- TimeGan: Combines unsupervised GAN with a supervised AR model and aims to generate time series with preserved temporal dynamics. I also don't understand what that means but imagine an auto-encoder with the encoded data in the middle of the architecture stolen to calculate G's loss. This turned out to be successful among state-of-the-art TS-GANs.

There are two other better continuous variants of TS-GANs but I'm gonna completely ignore them cause I don't understand their complexity. TBH I already don't understand a lot of what I've talked about. 

##   HOW TO FIX THIS SHIT?

Look, I yapped about temporal dynamics a lot in this article, but why is this a big problem? isn't there a way to fix this. Problem with time-series data is the fact that they tend to get very large and they make the adversarial learning space very high-dimensional. Meaning the long-term dynamics are very large for model to process and keep therefore making the learning process a pain in the lower back. The way to fix this is an Embedding Network.

Isn't that familiar?! I just introduced the network that does it... COME ON! TimeGAN.

##   TIME-GAN: INTRO

Your voice is a time-series data. Also your mom's voice! but there are two kind of feature to the voice, Static features(S) that don't change over time like gender of your voice (You don't suddenly sound like a girl unless you got hit in the crutch) and Temporal features(X) like the way you pronounce a word (You may say hello differently next time).

Our goal is to learn a distribution density like p'(S, X) that best immitates p(S, X) a.k.a the main dataset's distribution. depending on the lengths and dimensionality of data this can be difficult to optimize in the original GAN, so we add a conditional term to approximate better -> p(X[t] | S, X[1:t-1]). 

This concludes in two section of network: one relies on the presence of a perfect adversary and the second relies on the presence of ground truth.

##   DIGGING DEEPER!

TimeGAN has 4 sub-network: 

1. Embedding Function (Encoder)
2. Recovery Function (Decoder) 
3. Sequence Generator (G) 
4. Sequence Discriminator (D)

So your network learns how to save memory optimally, learn representations optimally, remember them optimally and then create things with those encoded representations all simuntaneously... it's like bethoven with his memory on stroids, he is already great at creating stuff but his memory helps him make richer musics with handing him not only the latest but the general patterns that makes great musics.

###  EMBEDDING & RECOVERY FUNCTION

There are two spaces here we should talk about the fearure space and the latent space, latent space is between the bottlenecks of an auto-encoder, in that space the only thing we find is the underlying dynamics. The meaning, The pattern, The essence of what makes data the way it is. Now Embedding function takes the static and temporal features to deep water (Khabib mentioned) and lets them discover their real selves. Embedding function in this model is a recurrent neural network but they can differ and Recovery has to only get our features out of the deep water and back to reality.

Now those embeddings, in between the bottlenecks are gonna be fed as the reality to the Discriminator, cause they are reality but compressed and NO BS.
