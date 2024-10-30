---
layout: post
title: Why Use GANs for Time Series?!
categories: [Deep Learning]
tags: [Deep Learning]
---
>I'm really bad at articulating stuff I learn and people mostly misunderstand me or they think I know nothing, so I decided to write stuff I learned about DL here and maybe this will help me articulate and understand things better.

So why use GANs for Time Series when there are plenty of good other alternatives like RNNs, LSTMs and AutoRegression Models?

First arguement about Time Series forcasting with this models is that they are more like regression (indication) than generation (prediction). In AutoRegression Models we keep predicting next timestamp as a function of previous timestamps therefore we only care about temporary dynamics of our dataset when we're iterating over each cell of the sequence (even LSTMs only care as far as their long-term memory or Cell state allows them).

The second arguement against them is that they introduce no randomness to prediction, they are not a one-to-many solution; meaning that for a point in a time-series there are not many options to predict but the one your model indicated from past. The fact that Time Series data have the uncertainity as their nature (unlike images or natural language) should also be embeded in the models that are analysing them and therefore why I suggest GANs.
