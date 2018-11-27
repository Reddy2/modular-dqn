# modular-dqn
This is an attempt to modularize many of the most effective extensions of the DQN algorithm so they can easily be mixed together.  The code is written in TensorFlow.  The goal is to easily test mixtures of extensions/algorithms like in the [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) paper, as well as to produce simple implementations of these algorithms for learning purposes.  See the examples folder to see how the module is used.  If you are looking for a more production ready library with many more features then try [TensorForce](https://github.com/reinforceio/tensorforce)

## Features
- [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- [Double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf)
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
  - Proporitional variant (tends to be favorable amongst most papers)
  - Rank-Based variant (Not yet optimized with precomputed segments)
- [Dueling Networks](https://arxiv.org/pdf/1511.06581.pdf)
- Multi-Step Learning
- [Distributional RL](https://arxiv.org/pdf/1707.06887.pdf) (C51)
- [Noisy Nets](https://arxiv.org/pdf/1706.10295.pdf)
- [Quantile Regression](https://arxiv.org/pdf/1710.10044.pdf) (QR-DQN)
- [NAF](https://arxiv.org/pdf/1603.00748.pdf) (Algorithm 1 only)

## What's missing
- [Frame skipping](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)
  - Because of this I have not yet added easy support for convolutional layers.  This can easily be done by using a custom network or by updating the code
  - This can probably easily be supported by using OpenAI Baseline's frame skipping code
- A few algorithms need to add support for gradient clipping and reward clipping
- A better way to save/load a model
- A logging system/TensorBoard support
- Documentation
- Probably a lot of things I'm not even aware of !

## In the future
- I hope to add the following:
  - [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf) (HER)
  - [Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf) (RND)
- Add [Horovod](https://github.com/uber/horovod) support for distributed training
- If possible to modularize, I would like to extend this module to include policy gradient methods such as PPO, rather than just DQN  
- I would like to add support for PyTorch

## Disclaimer
I don't have proof that any algorithm's implementation is correct.  In implementing an algorithm I do the following:
- Read the paper and any blogs on the paper and/or existing implementations I can find
- Write my own implementation of the algorithm
- Do basic math checks.  For example, making sure the algorithm gives a proper probability distribution
- Ensure the algorithm can gain a very high score on a basic OpenAI gym enviornment like CartPole or MountainCar
  - This sometimes requires a few tries, as these algorithms seem to be highly reliant on initial conditions (as seems to be reported by many working in the field)

In addition, the API will most likely change.  As more algorithms are added, sometimes a new level of abstraction is needed to keep the code modular (or simply a better way is found)

## To the learner
If you are trying to learn how to implement these algorithms and are having a hard time, then you are not alone.  Many of those who implement the main RL libraries seem to agree with this (See [Section 2](https://github.com/reinforceio/tensorforce/blob/master/FAQ.md)).  [Many implementations have glaring errors](https://github.com/devsisters/DQN-tensorflow/issues/16) due to ambiguous language like 'error clipping' or misunderstandings (this library itself may have many -- see the disclaimer).  There is not yet a standard textbook in the field.  Papers sometimes leave out/hop around important things like frame skipping or an edge case which causes loss of probability mass.  Sometimes the math seems to make no sense or skips way too many steps.  Sometimes trying to understand everything at once is overwhelming. Vectorizing something in multiple dimensions can have your head smashing into things in multiple dimensions.   Once you are finally convinced that your implementation works it may not replicate the results of a paper, take many tries to work, or work and then suddenly start doing poorly again (these results are normal).  Don't be discourged.  Even OpenAI recognize the learning curve and have just released Spinning Up in an attempt to help.  Here are some resources that have helped me along the way:
- [OpenAI Baselines](https://github.com/openai/baselines) (This is where I started from)
- [TensorForce](https://github.com/reinforceio/tensorforce) (Also check out their blog)
- [Reinforcement Learning](http://incompleteideas.net/book/bookdraft2017nov5.pdf) (Sutton, Barto 2017)
- [OpenAI Spinning Up](https://spinningup.openai.com) (I wish I had this in the beginning!)
- Blog posts.  These can be extremely helpful for understanding a paper
- Other people's implementations are of course wonderful.  But be wary that they are correct/complete (including within this module!)
