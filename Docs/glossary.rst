Glossary
********

.. Admonition:: Todos

    * For any given algorithm entry, make sure you provide a simple, pseudocode implementation as well.
    * Really need to consider the organization of the glossary--once you get enough content. It would probably be easier to categorize by "subject" or "field" than by alphabet.

.. contents:: Table of Contents
    :depth: 2
.. section-numbering::


.. a:

ant colony optimization (ACO)
=============================
An optimization algorithm inspired by the swarm intelligence of social ants using pheremone as a chemical messenger. First published in Marco Dorigo's phD thesis in 1992.


------


.. b:

bat algorithm (BA)
==================
**EXPAND** A continuous optimization algorithm based on the echolocation behavior of microbats. (2010 Yang, XS)

Bees
====
Algorithms inspired by the behavior of bees

bees algorithm
--------------
**TODO** 2005 D.T. Pham

artificial bee colony (ABC)
---------------------------
**TODO** 2005 Karaboga


honeybee algorithm
------------------
**TODO** 2004 Sunil Nakrani

virtual bee algorithm
---------------------
**TODO** 2005 Xin-She Yang

------


.. c:

categorical cross-entropy loss
==============================
The categorical cross-entropy loss is also known as the negative log likelihood. It is a popular loss function for categorization problems and measures the similarity between two probability distributions, typically the true labels and the predicted labels. It is given by ``L = -sum(y * log(y_prediction))`` where ``y`` is the probability distribution of true labels (typically a one-hot vector) and ``y_prediction`` is the probability distribution of the predicted labels, often coming from a softmax. [3]_

cuckoo search (CS)
==================
**TODO** 2009 Xin-She Yang

------

.. d:

deep belief network (DBN)
=========================
DBNs are a type of probabilistic graphical model that learn a hierarchical representation of the data in an unsupervised manner. DBNs consist of multiple hidden layers with connections between neurons in each successive pair of layers. DBNs are built by stacking multiple RBNs on top of each other and training them one by one. [3]_

differential evolution (DE)
===========================
**EXPAND**. A vector-based evolutionary algorithm published in 1996.

------

.. e:

exploding gradient problem
==========================
The exploding gradient problem is the opposite of the vanishing gradient problem. In deep neural networks gradients may explode during backpropagation, resulting number overflows. A common technique to deal with exploding gradients is to perform gradient clipping. [3]_

------

.. f:

firefly algorithm (FA)
======================
**TODO** 2007 Xin-She Yang

flower pollination algorithm
============================
**TODO** 2012 Xin-She Yang

------

.. g:

gradient clipping
=================
Gradient clipping is a technique to prevent exploding gradients in very deep networks, typically recurrent neural networks. There exist various ways to perform gradient clipping, but the a common one is to normalize the gradients of a parameter vector when its L2 norm exceeds a certain threshold according to ``new_gradients = gradients * threshold / l2_norm(gradients)`` [3]_

------

.. h:

harmony search (HS)
===================
**EXPAND** 2001 Zong Woo Geem

honeybee algorithm
==================
**see bees**

virtual bee algorithm
---------------------
placeholder

highway layer
=============
A Highway Layer is a type of Neural Network layer that uses a gating mechanism to control the information flow through a layer. Stacking multiple Highway Layers allows for training of very deep networks. Highway Layers work by learning a gating function that chooses which parts of the inputs to pass through and which parts to pass through a transformation function, such as a standard affine layer for example. The basic formulation of a Highway Layer is ``T * h(x) + (1 - T) * x``, where ``T`` is the learned gating function with values between 0 and 1, ``h(x)`` is an arbitrary input transformation and ``x`` is the input. Note that all of these must have the same size. See `Highway Networks <http://arxiv.org/abs/1505.00387>`_  [3]_

------

.. i:

------

.. j:

------

.. k:

------

.. l:

------

.. m:

------

.. n:

neural machine translation (NMT)
================================
An NMT system uses Neural Networks to translate between languages, such as English and French. NMT systems can be trained end-to-end using bilingual corpora, which differs from traditional Machine Translation systems that require hand-crafted features and engineering. NMT systems are typically implemented using encoder and decoder recurrent neural networks that encode a source sentence and produce a target sentence, respectively. [3]_

Neural Turing machine (NTM)
===========================
NMTs are Neural Network architectures that can infer simple algorithms from examples. For example, a NTM may learn a sorting algorithm through example inputs and outputs. NTMs typically learn some form of memory and attention mechanism to deal with state during program execution. [3]_

No Free Lunch Theorem
=====================
Published in 1997, the theorem states if algorithm A performs better than algorithm B for some optimization functions, then B will outperform A for other functions. ie, if averaged over all possible function space, both algorithms A and B will perform equally well. Alternatively, no universally better algorithms exist. [Yang2014]_

By NFLT, there is no universally better optimization algorithm. However, research can be devoted to finding the most efficient algorithm for a given set of problems.

------

.. o:


------

.. p:

particle swarm optimization (PSO)
=================================
Optimization algorithm inspired by swarm intelligence of fish and birds and even by human behavior. The multiple agents, called *particles*, swarm around the search space, starting from some initial random guess. The swarm communicates the current best guess and shares the global best so as to focus on the quality solutions.

Since it's publication in 1995, there have been about 20 different variants of PSO techniques, which have been applied to almost all areas of challenging optimization problems, and there is strong evidence that PSO is better than traditional search algorithms and even better than GA for many types of problems. [Yang2014]_

PSO variants
------------
placeholder


------

.. q:

------

.. r:

recursive neural network
========================
Recursive Neural Networks are a generalization of Recurrent Neural Networks to a tree-like structure. The same weights are applied at each recursion. Just like RNNs, Recursive Neural Networks can be trained end-to-end using backpropagation. While it is possible to learn the tree structure as part of the optimization problem, Recursive Neural Networks are often applied to problem that already have a predefined structure, like a parse tree in Natural Language Processing. [3]_

ResNet
======
Deep Residual Networks won the ILSVRC 2015 challenge. These networks work by introducing shortcut connection across stacks of layers, allowing the optimizer to learn “easier” residual mappings instead of the more complicated original mappings. These shortcut connections are similar to Highway Layers, but they are data-independent and don’t introduce additional parameters or training complexity. ResNets achieved a 3.57% error rate on the ImageNet test set. [3]_

restricted Boltzmann machine (RBM)
==================================
RBMs are a type of probabilistic graphical model that can be interpreted as a stochastic artificial neural network. RBNs learn a representation of the data in an unsupervised manner. An RBN consists of visible and hidden layer, and connections between binary neurons in each of these layers. RBNs can be efficiently trained using Contrastive Divergence, an approximation of gradient descent. [3]_

------

.. s:

------

simulated annealing (SA)
========================
Metaheuristic inspired by the annealing process of metals. It is a trajectory-based search algorithm, starting with an initial guess solution at a high temperature and gradually cooling down the system. A move or new solution is accepted if it is better; otherwise, it is accepted with a probability, allowing it to escape any local optima. It is then expected that if the system is cooled down slowly enough, the global optimal solution can be reached. [Yang2014]_

softmax
=======
The softmax function is typically used to convert a vector of raw scores into class probabilities at the output layer of a Neural Network used for classification. It normalizes the scores by exponentiating and dividing by a normalization constant. If we are dealing with a large number of classes, a large vocabulary in Machine Translation for example, the normalization constant is expensive to compute. There exist various alternatives to make the computation more efficient, including Hierarchical Softmax or using a sampling-based loss such as NCE. [3]_

Swarm Intelligence (SI)
=======================
**TODO** Expand (this is a major domain/umbrella for many NIH)

.. t:

------

.. u:

------

.. v:

------

.. w:

------

.. x:

------

vanishing gradient
==================
The vanishing gradient problem arises in very deep Neural Networks, typically Recurrent Neural Networks, that use activation functions whose gradients tend to be small (in the range of 0 from 1). Because these small gradients are multiplied during backpropagation, they tend to “vanish” throughout the layers, preventing the network from learning long-range dependencies. Common ways to counter this problem is to use activation functions like ReLUs that do not suffer from small gradients, or use architectures like LSTMs that explicitly combat vanishing gradients. The opposite of this problem is called the exploding gradient problem. [3]_


.. y:

------

.. z:

------


References
==========


.. [Yang2014] Yang, Xin-She. (2014). Nature-Inspired Optimization Algorithms. `Full-text PDF <https://www.researchgate.net/publication/263171713_Nature-Inspired_Optimization_Algorithms>`_


.. [3] From `Denny Britz' <https://twitter.com/dennybritz/>`_ `Deep Learning Glossary <http://www.wildml.com/deep-learning-glossary/>`_, acc. 2018-11-22
