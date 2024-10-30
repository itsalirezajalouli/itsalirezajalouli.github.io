---
layout: post
title: Why Use GANs for Time Series?!
categories: [Deep Learning]
tags: [Deep Learning]
---
>I'm already bad at articulating stuff I learn and people mostly misunderstand me or think I know ****, so I decided to write stuff I learned lately and maybe this will help me understand them better too, or even be helpful to others.

##   INTRO

So why use GANs for Time Series when there are plenty of good other alternatives like RNNs, LSTMs and AutoRegression Models? the loop-like structure of them make them perfect for sequential data. doesn't it?

First arguement about Time Series forcasting with this models is that they are more like regression (indication) than generation (prediction). In AutoRegression Models we keep predicting next timestamp as a function of previous timestamps therefore we only care about temporary dynamics of our dataset when we're iterating over each cell of the sequence (even LSTMs only care as far as their long-term memory or Cell state allows them).

The second arguement against them is that they introduce no randomness to prediction, they are not a one-to-many solution; meaning that for a point in a time-series there are not many options to predict but the one your model indicated from past. The fact that Time Series data have the uncertainity as their nature (unlike images or natural language) should also be embeded in the models that are analysing them and therefore why some people suggest GANs.

GANs aren't that perfect either, they are inherently unstable models. They have problems like non-convergence and vanishing gradients (meaning Discriminator is way more successful than Generator so G keeps loosing and thus learns nothing), and mode collapses (when G learns to fool D with only one class of the data and stops learning more and keeps sending that variant repeatedly to the D) which are not completely solved yet. Contrary to image-based GANs there is no unified metric for TS GANs and that makes their performanse evaluation subjective. GANs for TS has been mentioned by a lot of papers in past few years but now it's getting more attention and becoming a growing body of research.

##   VARIANTS OF TIME SERIES BASED GANS

Time Series GANs come in two variants: Discrete and Continuous, GANs struggle with discrete data because there is zero gradients almost everywhere due to the nature of the data, so you can't use backpropogation alone. here I list some interesting TS-GAN models which performed competitively: 

#### Discrete Variants:

- SeqGAN: This one was made to fix the discrete backprop problem, here G is a LSTM, D is a CNN and G's gradient gets updated by a policy gradient and Monte Carlo Search 
- QuantGAN: This one came along to capture those long-term dependencies (long long long-term memories) which LSTMs couldn't preserve, both G & D are temporal convolutional networks with skip connections. authors claim they can outperform other conventional mathematical finance models.

#### Continuous Variants:
