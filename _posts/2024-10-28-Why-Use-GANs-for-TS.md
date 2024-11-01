---
layout: post
title: Why Use GANs for Time Series?!
categories: [Deep Learning]
tags: [Deep Learning]
---
>I know I'm bad at remembering and articulating stuff I learned so I decided to write stuff here, if I suck at grammar or suffer from lack of vocabulary variety... DEAL WITH IT, it's my blog!

##   INTRO

So why use GANs for Time Series when there are plenty of good alternatives like RNNs, LSTMs and AutoRegression Models? The loop-like structure of these makes them perfect for sequential data, doesn't it?

##   NOPE!

The first argument about Time Series forecasting with these models is that they are more like regression (indication) than generation (prediction). In AutoRegression Models, we keep predicting the next timestamp as a function of previous timestamps; therefore, we only care about temporary dynamics of our dataset when we're iterating over each cell of the sequence (even LSTMs only care as far as their long-term memory or Cell state allows them).

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

##   GANS AS DATA AUGMENTATION
I don't know why I added this section - nobody should use them for data augmentation unless they have big bags of data, but if you have TS data like TTS and music generators, kinda use this to learn better. It's funny though, cause one of the uni's profs I've worked with has GAN augmentation fetish, not gonna say who... Nik....
