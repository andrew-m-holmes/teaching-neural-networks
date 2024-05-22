# Teaching Neural Networks

![header](https://github.com/andrew-m-holmes/teaching-networks/blob/main/images/images_modern%20Huggies_Acedemic%20Huggy.png?raw=true)

## Introduction
---
As of today, machine learning (ML) is an exponentially growing field thanks to the popularity and wide spread use of [large language models (LLM)s]()—a specific class of neural networks synonymous to the [transformer architecture](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)). They've been adopted into many applications, some of which you're probably familiar with like [ChatGPT]() and [Copilot](). Altogether, these models have proven to be valuable assets in aiding learning, productivity, and business. But, like humans, they don't come out of the box with all the knowledge and information in the world. Just as we learn, they must go through a learning process which adapts their behavior to make them practical and viable tools for different domains. 

This learning process that the majority of neural networks go through today is called *training*. The algorithm predominantly used to train these models is **gradient descent**. In a nutshell, gradient descent is an optimization algorithm that tries to incrementally improve the performance of a neural network as it's trained over data. In this blog, we're going to unpack gradient descent, learn about its variants, and explore popular **optimizers** that serve to make gradient descent more effective.

## What Do We Want To Solve?
---
Because of open source companies like [Hugging Face](), the amount of data available for a variety of domains has become ubiquitous. Essentially, we want to leverage or interpret this data for some particular purpose. Most often we're trying to find a correlation between the inputs and outputs of a dataset which has proven to be extremely useful for a wide range of tasks.

For example, we might want to determine if a tumor is benign or malignant from radiology imaging, or determine the object(s) captured from the cameras of a self driving car so it react appropriately. With LLMs, we want to take in a chunk of text, semantically understand it, and give a reasonable response back based on the context. The common theme amongst these tasks is we want to accurately predict outputs from inputs. Neural networks enable us to achieve this objective in the form of *supervised learning*, or learning based on predicting the expected value of outputs from inputs given we know their ground truth.

<h3 style="text-align: center;">GPT-2 Attention Mechanism</h3>

![attention-mechanism](https://github.com/andrew-m-holmes/teaching-networks/blob/main/scripts/attention-mechanism.gif?raw=true)

> Above is the attention mechanism of the twelve attention heads found in the first layer of GPT-2—one the first large scale generative transformers. With the distinction made by color amongst the heads, how tokens (words/subwords) attend to one another (i.e. place importance on) other tokens can be interpreted based on the opacity of each head's color. From the depiction, it can be hypothesized that the model learned to correlate "pancakes", "eggs", and "oatmeal" with "breakfast", as they attend to it more than other tokens less related to the items. This ability to learn, evidently, is possible thanks to gradient descent. (Implementation courtesy of [bertviz]()).

## Understanding Our Toolbox
---

# [NEURAL NETWORK MANIM]

Neural networks—the layered inter-connected web of [artificial neurons](https://en.wikipedia.org/wiki/Artificial_neuron)—are really just funcitons \\(f: \R^n \to \R^c \\) parameterized by a set of values \\( \theta \in \R^d \\), where \\( \theta \\) is the parameters of the network and \\( d \\) is how many parameters the network has. To compliment, each input from a dataset is denoted \\( x \in \R^n \\) and the corresponding label \\( y \in \R^c \\), where \\( n \\) represents the number of features for each input and \\( c \\) correlates to the number of classes/output activations for a supervised task. Additionally, a network's prediction computed via forward propagation is denoted \\(\hat{y} \in \R^c \\) such that

$$\hat{y} = f(x; \theta)$$

Continuing, we're also given an objective function that generates a positive real-valued scalar measuring the *loss* (or error) between a network's prediction \\( \hat{y} \\) and its correct label \\( y \\). This objective function is called the *loss function*; it's pretty much a numerical way to determine how good or bad a neural network is at accurately predicting outputs from inputs, provided the correct labels.

As an example for better interpretation, we'd use [cross-entropy loss](https://en.wikipedia.org/wiki/Cross-entropy) to measure the error between the predicted probabilities of our model \\( \hat{y} \\) and their correct labeling \\( y \\) (in the form of a [one-hot vector](https://en.wikipedia.org/wiki/One-hot#:~:text=Natural%20language%20processing%5B,the%20pre%2Dprocessing%20part.)) for some classification task. We'd define the cross-entropy loss \\( \ell: \R^c \to \R_+ \\) as 

$$ \ell(\hat{y}, y) = -\displaystyle\sum^c_{j=1} y_jlog\hat{y}_j$$

where \\( \hat{y}_j \\) is the probability of the input \\( x \\) being the \\( j\text{-th} \\) class and \\( y_j \\) is \\( 0 \\) or \\( 1 \\) denoting whether \\( x \\) is or isn't the \\( j\text{-th} \\) class respectively.

Now, this loss only functions on one prediction and label but we're usually trying to understand the model's performance over the entire dataset with inputs \\( X \in \R^{m \times n}\\) and labels \\( Y \in \R^{m \times c} \\), where \\( m \\) denotes the number of training examples in the entire dataset. Now, our network's predictions \\( \hat{Y} \in \R^{m \times c} \\) are described as

$$ \hat{Y} = f(X; \theta) $$

To account for this, we compute the average loss over the entire dataset such that

$$J(\hat{Y}, Y) = \frac{1}{m}\displaystyle\sum^m_{i=1}\ell(\hat{y}^{(i)}, y^{(i)})$$

where \\( \hat{y}^{(i)} \\) and \\( y^{(i)} \\) are the prediction and label of the \\( i\text{-th} \\) input \\( x^{(i)} \\) respectively, and  
 \\( J: \R^{m \times c} \to \R_+ \\) is a *cost function* which (intuitively) indicates the "cost" of our networks parameters on the loss.

> As an aside, cost and loss [function] are often (reasonbly so) used interchangably as the former is just an average of the latter. For simplicity, and ultimately laziness, I'll refer to the output of \\( J \\) as the loss. Additionally, it's typical to simplify the loss as \\( J(\theta) \\)—a parameterization of the neural network's parameters—since the dataset remains fixed, while \\( \theta \\) is the only argument affecting \\( J \\)'s output. For brevity, I will use this shorthand unless clarification is required.

The last aspect we must discuss is \\( \nabla_{\theta}J(\theta) \\), or the *gradient* of the loss with respect to (w.r.t.) our network's parameters. The gradient is computed via [backpropagation]()—the algorithm leveraging the [chain rule]() to propagate the partial derivatives of the loss w.r.t. to each parameter starting from the output layer, traversing through the intermediate hidden layer(s), and ending with the input layer.  

$$ \nabla_{\theta}J(\theta) = \begin{bmatrix} \frac{\partial J}{\partial \theta_1} \\ \vdots \\ \frac{\partial J}{\partial \theta_d} \end{bmatrix}$$

 \\( \nabla_{\theta}J(\theta) \\) is a vector in \\( \R^d \\) that encodes the direction of *steepest ascent* of \\( J \\) at the point \\( J(\theta) \\). At a high level, the gradient is significant because it gives us an estimation for what direction our parameters are moving along the *surface* of the loss.

# [PLOT LOSS PROJECTED TO 3D SPACE]

 With enough background, notation, and clarification let's begin to unpack gradient descent.

 ## Gradient Descent
 ---

As mentioned before, gradient descent is an optimization algorithm whose main purpose is to iteratively reduce the loss of an objective function parameterized by a neural network's parameters. Essentially, gradient descent wants to reach the global minimum of the loss, granted the loss is convex; otherwise, it tries to reach a local minimum since the loss (typically) is non-convex. 

The trick in how it achieves this is by leveraging the gradients from backpropagation. Since the gradients \\( \nabla_{\theta}J(
\theta) \\) are an estimation to where the loss is increasing the most (steepest ascent) from the network's parameters \\( \theta \\), we'd shift the parameters in the direction opposite the gradient (i.e. the negative direction) in order to decrease the loss on a subsequent training step*.

> *A training step is a fixed point in time in which we change a network's parameters from one state of values into another using their gradients on any given iteration of gradient descent.

Systematically, we:

1. Forward propagate inputs through our neural network to generate the network's predictions.

2. Compute the loss between the network's predictions and their correct labels using a loss function.

3. Compute the gradients of the loss w.r.t. the network's parameters using backpropagation.

4. *Step* (i.e. adjust the network's parameters) in the direction opposite the gradient.

5. Repeat steps 1-4 until the model *converges*; this is where the gradients are near zero and the loss no longer improves on subsequent training steps.

In this process, we're repeatedly descending from some point of the loss \\( J(\theta) \\) using gradients \\( \nabla_{\theta}J(\theta) \\), and hence the name gradient descent. The *update rule*—how we take a step for each iteration of gradient descent—is formulated by

$$\theta_{t + 1} = \theta_t - \eta \nabla_{\theta}J(\theta_t)$$

where \\( t \\) represents the current time step during training, and \\( \eta \\) is a [hyperparameter]() called the *learning rate* which controls how big of a step we take when optimizing the network's parameters. With this additional term, step 4 should be

> 4. *Step* (i.e. adjust the network's parameters) in the direction opposite the gradient scaled by the learning rate.

The learning rate actually plays a significant role during training. Too big of a value and the network's parameters will oscillate back and forth in a suboptimal region of the loss, or will *diverge* entirely. On the opposite end of the spectrum, if you choose a learning rate that's too small, gradient descent will likely converge at the expense of taking a significant amount of time to do so. 

# [Oscilating, Diverging, and Slow Learning Rate]

Choosing a learning rate in itself is considered an "*art*", but we'll learn later in this blog that this burden is lifted. For now, let's focus on the three variants of gradient descent that determine when a step is taken: **batch gradient**, **stochastic gradient descent**, and **mini-batch gradient descent**.

## Batch Gradient Descent
---
We've somewhat already discussed batch (or *vanilla*) gradient descent, but it's a variant of gradient descent that updates the network's parameters only once it's seen the entire dataset. More specifically, we must pass all the examples of the dataset through the network, compute the loss from the generated predictions, and finally use the gradients from backpropagation to step. The update rule is defined as 

$$\theta_{t + 1} = \theta_t - \nabla_{\theta}J(X, Y; \theta_t)$$

where \\( X \\) and \\( Y \\) are the inputs and labels of a dataset respectively from before. Here, it's just more explicit that we're passing the entire dataset when deriving the gradients w.r.t. to the network's parameters \\( \nabla_{\theta}J(\theta) \\).

<h3 style="text-align: center;">Batch Gradient Descent on MNIST</h3>

![batch-gradient descent plot](https://github.com/andrew-m-holmes/teaching-networks/blob/main/images/batch_metrics.png?raw=true)

> Depicted above is the loss plotted during training of a 4 layer neural network on a subset (10,000 examples) of the MNIST handwritten digit dataset. The model was trained for 100 epochs (i.e. the amount of times the model sees the entire dataset) using cross entropy loss, and was tested on a subset of test examples (1,000 examples) using the same objective. From the plot, batch gradient descent is a little stagnant at first, but smoothens as it begins to converge to an optimum.

With this approach, gradient descent will always converge to a global minimum if the loss is convex and a local minimum otherwise. However, there's quite a few caveats. For one, we must pass the entire dataset to our model and loss function in one go; this is a cost in the computational departments of both speed* and memory. Furthermore, some examples are wasted on a given training step. Since the model will generate similar predictions for similar examples, their gradients will be roughly identical, and thus won't add any additional information to help the model learn.

> *The environment in which computation is slow is under the assumption that no form of efficient paralleization is being used to generate model outputs. This environment is not as prevalent because of libraries like [numpy](https://numpy.org/doc/stable/index.html) and [PyTorch](https://pytorch.org/)—commonly used for ML workloads—which use [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) and/or [CUDA](https://en.wikipedia.org/wiki/CUDA) to process data in parallel for incredibly fast compute. In spite of this, I felt it necessary to include this claim for conceptual reasons.

# [CUDA OOM ERROR]
Lastly, this variant of gradient does not support *online-learning*—the ability to update the model after each new example seen in production. Some models deployed online, like [recommendation engines](), should improve overtime based on individual user interactions, but with batch gradient descent, this cannot be achieved which can degrade the performance of an ML application as the distribution of data shifts.

To wrap up, batch gradient descent is usually not used for the issues above, however, it's preferred when the dataset is relatively small—no more than 10,000 examples. The problems explained above can be addressed in the next variant of gradient descent.

## Stochastic Gradient Descent
---

Stochastic gradient descent (SGD) is a variant of gradient descent that updates a netowork's parameters for each example seen during training. Primarily, we compute the loss and the gradients of the loss w.r.t. the network's parameters for just one example, and then update. The update rule for SGD is 

$$\theta_{t + 1} = \theta_t - \eta \nabla_{\theta}J(x^{(i)}, y^{(i)}; \theta_t)$$

where \\( x^{(i)} \\) and \\( y^{(i)} \\) are the \\( i\text{-th}\\) input and label from a dataset, however, they're in \\( \R^{1 \times n} \\) and \\( \R^{1 \times c} \\) (respectively) to comply with the definition of our loss \\( J \\).

<h3 style="text-align: center;">Stochastic Gradient Descent on MNIST</h3>

![stochastic gradient descent plot](https://github.com/andrew-m-holmes/teaching-networks/blob/main/images/stochastic_metrics.png?raw=true)

> Represented above is the same training and evaluation procedures elaborated in the batch gradient descent depiction, however, using SGD. Unlike batch gradient descent, SGD is *"spikey"* as it trains and struggles to converge on the test loss—likely due to the frequency of stochastic updates.

The main benefit of this variant are the drawbacks of batch gradient descent corrected. It's much faster* and memory efficient because we don't have to feed nor store the entire dataset in memory before a training step is taken. Also, we're no longer wasting examples, in turn gradients because each gradient is specialized to a particular example and state of the model. This makes the updates more informative and also gives them the ability to get out of suboptimal minimums. Lastly, there's support for online learning as this variant is specialized to make parameter updates for each example.

> *As explained before, this goes under the assumption that parellization was not used for efficient computations (explained in the batch gradient descent section).

Like before, however, there's caveats. For one, every update is stochastic—random to say the least. This variance in gradients can make it difficult to interpret if the network is improving overtime since the loss has random *"spikes"* in different directions. Lastly, the parameters tend to osciallate around a minimum since the training steps are taken for each example, requiring the need to manually change the learning rate to converge.

# [VARIANCE OF GRADIENTS OR OSCILLATION AROUND MINIMUM]

Altogether, SGD is a useful form of gradient descent, especially when online learning is required and the features of examples aren't as sparse. There's still room for improvement, in which we can bridge the benefits of batch gradient descent and SGD to give us a stable middle ground for training.

## Mini-batch Gradient Descent
---

Mini-batch gradient descent is the last variant of gradient descent that takes a training step for each *mini-batch*. A mini-batch is technically any subset of examples from the dataset defined by the *batch size* hyperparameter—simply the number of examples in each mini-batch. The batch size is almost always a power of 2 (for efficient memory accessing), usually no smaller than 16 and no larger than 512. The update rule for mini-batch gradient descent is defined by

$$\theta_{t + 1} = \theta_t - \eta \nabla_{\theta}J(X^{(i:i + b)}, Y^{(i:i + b)}; \theta_t)$$

where \\( i \\) is some index bewteen 1 and the start of the last mini-batch, and \\( b \\) the batch size hyperparameter. 

For better conceptualization, let \\( b = 64 \\) and \\( m = 1000 \\). The first mini-batch (exclusive of the last index) would be \\( X^{(1:65)} \\) and \\( Y^{(1:65)} \\). Similarly, the second would be \\( X^{(65:129)} \\) and \\( Y^{(65:129)} \\). The last—with only 40 examples—would be \\( X^{(961:1001)} \\) and \\( Y^{(961:1001)} \\) because the batch size doesn't divide the dataset into indentically sized mini-batches. In choosing \\( b = m \\), we'd have batch gradient descent, and for \\( b = 1 \\), we'd have SGD.

<h3 style="text-align: center;">Mini-batch Gradient Descent on MNIST</h3>

![min-batch gradient descent plot](https://github.com/andrew-m-holmes/teaching-networks/blob/main/images/mini_batch_metrics.png?raw=true)

> In the same constraints as the last two loss plots, depicted above is mini-batch gradient descent trained and evaluated on a subset of MNIST training and testing examples. Each mini-batch had 512 examples with exception to the last batch (272 examples). It can be viewed that mini-batch moves along the surface of the loss in a more controlled manner, and eventually converges towards a good minimum.

The benefits of mini-batch gradient descent come from leveraging batch and stochastic gradient descent. Assuming an appropriate batch size was chosen, the training steps don't incur as much computational or memory overhead because the mini-batches are relatively small compared to the size of the entire dataset. In addition, we're not wasting examples because each update has a level of stochasticity. In a sense, we're capturing just enough information to learn but not so much that the information becomes redundant.

Continuing, the training steps aren't highly variant because more examples are incorporated per update, leading to more gradual steps taken towards the minimum. Lastly, the majority of machine learning frameworks develop kernels (software for hardware) specifically designed to handle batched computations, giving us *"more bank for our buck"* for computations.

# [TABLE OF WHAT VARIANT OF GRADIENT DESCENT TO USE BASED ON SCENARIO]

When it comes to training a model, mini-batch gradient descent is typically preferred for the reasons above. However, there are certain situations where you might prefer batch or stochastic over mini-batch gradient descent due to the constraints of the problem. 

> Note: mini-batch gradient descent is commonly called stochastic gradient descent because it's still stochastic in nature. If you've read documentation of certain machine learning libraries or papers, references to SGD is usually referring to mini-batch gradient descent unless explcitly stated otherwise. For the remainder of this blog, I will do the same.

## Optimizers
---
