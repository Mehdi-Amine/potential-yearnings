# Dismantling Neural Networks to Understand the Inner Workings with Math and Pytorch
## Motivation
As a child, you might have dismantled a toy in a moment of frenetic curiosity. You were drawn perhaps towards the source of the sound it made. Or perhaps it was a tempting colorful light from a diode that called you forth, moved your hands into cracking the plastic open.

Sometimes you may have felt deceived that the inside was nowhere close to what the shiny outside led you to imagine. I hope you have been lucky enough to open the right toys. Those filled with enough intricacies to make breaking them open worthwhile. Maybe you found a futuristic looking DC-motor. Or maybe a curious looking speaker with a strong magnet on its back that you tried on your fridge. I am sure it felt just right when you discovered what made your controller vibrate.

We are going to do exactly the same. We are  dismantling a neural network with math and with Pytorch. It will be worthwhile, and our toy won’t even break. Maybe you feel discouraged. That’s understandable. There are so many different and complex parts in a neural network. It is overwhelming. It is the rite of passage to a wiser state.

So to help ourselves we will need a reference, some kind of Polaris to ensure we are on the right course. The pre-built functionalities of Pytorch will be our Polaris. They will tell us the output we must get. And it will fall upon us to find the logic that will lead us to the correct output. If differentiations sound like forgotten strangers that you once might have been acquainted with, fret not! We will make introductions again and it will all be mighty jovial.  
I hope you will enjoy.

<br/><br/> 
## Linearity
The value of a neuron depends on its inputs, weights, and bias. To compute this value for all neurons in a layer, we calculate the dot product of the matrix of inputs with the matrix of weights, and we add the bias vector. We represent this concisely when we write:

$$z = x \cdot w^T + b$$

Conciseness in mathematical equations however, is achieved with abstraction of the inner workings. The price we pay for conciseness is making it harder to understand and mentally visualize the steps involved. And to be able to code and debug such intricate structures as Neural Networks we need both deep understanding and clear mental visualization. To that end, we favor  verbosity:

$$z = x_0*w_0 + x_1*w_1 + x_2*w_2 + b$$

Now the equation is grounded with constraints imposed by a specific case: one neuron, three inputs, three weights, and a bias. We have moved away from abstraction to something more concrete, something we can easily implement:

```python
import torch

x = torch.tensor([0.9, 0.5, 0.3])
w = torch.tensor([0.2, 0.1, 0.4])
b = torch.tensor([0.1])

z_v = x[0]*w[0] + x[1]*w[1] + x[2]*w[2] + b # Verbose z
z = x @ w + b # Concise z

print(f"Verbose z: {z_v} \nConcise z: {z}")
'''
Out:
Verbose z: tensor([0.4500]) 
Concise z: tensor([0.4500])
'''
```

To calculate $z$, we have moved forward from a layer of inputs to the next layer of neurons. When a neural network steps all the way forward through its layers and acquires knowledge, it needs to know how to go backwards to adjust its previous layers. We can achieve this backward propagation of knowledge through derivatives. Simply put, if we differentiate $z$ with respect to each of its parameters (the weights and the bias), we can get the values of the input layer $x$.

If you have forgotten how to differentiate, rest assured: you won’t be told to go brush up on an entire branch of calculus. We will recall differentiations rules as we need them. The partial derivative of $z$ with respect to a parameter tells you to consider that parameter as a variable, and all other parameters as constants. The derivative of a variable is equal to its coefficient. And the derivative of a constant is equal to zero:

$$\frac{\partial z}{\partial w_0} = (x_0*w_0)' + (x_1*w_1)' + (x_2*w_2)' + b' = x_0$$

Similarly, you can differentiate $z$ with respect to $w_1$, $w_2$, and $b$ (with $b$ having the invisible coefficient of 1). You will find that every partial derivative of $z$ is equal to the coefficient of the parameter with respect to which it is differentiated. With this in mind, we can use **Pytorch Autograd** to evaluate the correctness of our math.

```python
import torch

x = torch.tensor([0.9, 0.5, 0.3])
w = torch.tensor([0.2, 0.1, 0.4], requires_grad=True)  
b = torch.tensor([0.1], requires_grad=True)
# requires_grad=True tells Pytorch we are differentiating w.r.t the weights and bias

z = x @ w + b 
z.backward() # differentiates z and stores the derivatives in w.grad and b.grad

print(f"Partial derivatives: {w.grad} {b.grad}")
'''
Out:
Partial derivatives: tensor([0.9000, 0.5000, 0.3000]) tensor([1.])
'''
```
<br/><br/> 
## Non-Linearity
We introduce non-linearity with activation functions. This enables neural networks to be *universal function approximators*. There are various types of activations, each one fulfills a different purpose and produces a different effect. We will go through the formula and differentiation of $ReLU$, $Sigmoid$, and $Softmax$.

### ReLU
The Rectified Linear Unit function compares the value of a neuron with zero and outputs the maximum. We can think of ReLU labeling all nonpositive neurons as equally inactive.
$$ReLU(z) = max(0,z)$$

To implement our own ReLU, we could compare z with 0 and output whichever is greater. But the [clamp](https://pytorch.org/docs/stable/torch.html#torch.clamp) method provided in the Torch package can already do this for us. In Numpy, the equivalent function is called [clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html). The following code implements a clamp-based ReLU, before using Pytorch’s [relu](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu) to evaluate its output.


```python
import torch
import torch.nn.functional as F

#----------- Implementing the math -----------#

def relu(z):
  return torch.clamp(z, 0, None) # None specifies that we don't require an upper-bound

z = torch.tensor([[-0.2], [0.], [0.6]]) # Three neurons with different values

relu = relu(z)

#----------- Using Pytorch -----------#

torch_relu = F.relu(z)

#----------- Comparing outputs -----------#

print(f"Pytorch ReLU: \n{torch_relu} \nOur ReLU: \n{relu}")
'''
Out:
Pytorch ReLU: 
tensor([[0.0000],[0.0000],[0.6000]]) 
Our ReLU: 
tensor([[0.0000],[0.0000],[0.6000]])
'''
```

The differentiation of ReLU is straightforward:

$$ \begin{equation*} ReLU'(z)=\begin{cases}1, & \text{if}\ z>0 \\0, & \text{otherwise}\end{cases}\end{equation*} $$

- For all positive $z$, the output of $ReLU$ is $z$. Therefore the differentiation is the coefficient of $z$, which is equal to one.

- For all non-positive $z$, the output of $ReLU$ is equal to zero. Therefore the differentiation is also equal to zero.

Let’s translate our understanding into Python code. We will implement our own $ReLU’(z)$ before comparing it with the automatic differentiation of Pytorch’s ReLU.

```python
import torch
import torch.nn.functional as F

#----------- Implementing the math -----------#

def relu_prime(z):
  return torch.where(z>0, torch.tensor(1.), torch.tensor(0.))

z = torch.tensor([[-0.2], [0.6]], requires_grad=True)

relu_p = relu_prime(z)

#----------- Using Pytorch autograd -----------#

torch_relu = F.relu(z)

torch_relu.backward(torch.tensor([[1.], [1.]])) 

#----------- Comparing outputs -----------#

print(f"Pytorch ReLU': \n{z.grad} \nOur ReLU': \n{relu_p}")
'''
Out:
Pytorch ReLU': 
tensor([[0.],[1.]]) 
Our ReLU': 
tensor([[0.],[1.]])
'''
```

*Why are we giving a tensor of ones to backward()?*

`backward()` defaults to the case of being called on a single scalar and uses the default argument `torch.tensor(1.)` This was previously the case when we called `z.backward()`. Since `torch_relu` is not a single scalar we need to explicitly provide a tensor of ones equal in shape to `torch_relu`.

### Sigmoid

The sigmoid activation function produces the effect of mapping $z$ from $ℝ$ to the range $[0,1]$. When performing binary classification, we typically label instances belonging to the target class with the value $1$, and all else with the value $0$. We interpret the output of $sigmoid$ as the probability that an instance belongs to the target class.

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

**Quiz:** The task of a neural network is to perform binary classification. The output layer of this network consists of a single neuron with a sigmoid activation equal to $0.1$. Among the following interpretations, which one(s) are correct?

1. There is a $0.1$ probability that the instance belongs to class $1$, the target class.
2. There is a $0.1$ probability that the instance belongs to class $0$.
3. There is a $0.9$ probability that the instance belongs to to class $0$.

**Solution:** only 1 and 3 are correct. It is important to understand that a sigmoid-activated neuron with some output $p$, is implicitly giving an output of $1-p$ for the non-targeted class. It is also important to keep in mind that $p$ is the probability associated with the target class (usually labeled as $1$), while $1-p$ is the probability associated with the non-targeted class (usually labeled as $0$).

**Observe:** that the sum of $p$ and $(1-p)$ is equal to $1$. This seems too obvious to point out at this stage, but it will be useful for us to keep it in mind when we discuss $Softmax$.

Once again, we translate the math in Python then we check our results with the Pytorch implementation of $sigmoid$:

```python
import torch

#----------- Implementing the math -----------#

def sigmoid(z):
  return 1 / (1+torch.exp(-z))

z = torch.tensor([[2.], [-3.]]) # Two neurons with different values

sig = sigmoid(z) 

#----------- Using Pytorch -----------#

torch_sig = torch.sigmoid(z)

#----------- Comparing outputs -----------#

print(f"Pytorch Sigmoid: \n{torch_sig} \nOur Sigmoid: \n{sig}")
'''
Out:
Pytorch Sigmoid: 
tensor([[0.8808],[0.0474]]) 
Our Sigmoid: 
tensor([[0.8808],[0.0474]])
'''
```
<br/><br/> 
$$\sigma'(z) = \sigma(z) \cdot (1-\sigma(z))$$

There is something graceful about the differentiation of $sigmoid$. It does, take a sinuous path to reach its grace, but once we recall a few differentiation rules, we will have all what we need to saunter our way down the sinuous path.

{% include dismantling_imgs/sigmoid-diff@2x.png %}

Having understood how to differentiate $sigmoid$, we can now implement the math and evaluate it with Pytorch’s `Autograd`.

```python
import torch

#----------- Implementing the math -----------#

def sigmoid_prime(z):
  return sigmoid(z) * (1 - sigmoid(z))

z = torch.tensor([[2.], [-3.]], requires_grad=True)

sig_p = sigmoid_prime(z)

#----------- Using Pytorch autograd -----------#

torch_sig.backward(torch.tensor([[1.], [1.]]))

#----------- Comparing outputs -----------#

print(f"Pytorch Sigmoid': \n{z.grad} \nOur Sigmoid': \n{sig_p}")
'''
Out:
Pytorch Sigmoid': 
tensor([[0.1050],[0.0452]]) 
Our Sigmoid': 
tensor([[0.1050],[0.0452]])
'''
```

Nowadays, $ReLU$ has been widely adopted as a replacement for $sigmoid$. But $sigmoid$ is still lingering around, hiding under the name of its more generalized form: $Softmax$.

<br/><br/>
### Softmax
We think of $sigmoid$ for binary classification, and $softmax$ for multi-class classification. This association while correct, misleads many of us into thinking that $sigmoid$ and $softmax$ are two different functions. This is emphasized by the fact that when we look at the equations of $sigmoid$ and $softmax$, it does not seem like there is an apparent link between them.

{% include dismantling_imgs/sm/softmax@2x.png %}

*A softmax activated neuron is the exponential of its value dived by the sum of the exponentials of all other neurons sharing the same layer.*

Once again, the abstraction of the formula makes it anything but intuitive at first glance. An example will make it more concrete. We take a case of two output neurons, the first one $z_0$ outputs the probability that instances belong to a class labeled $0$, the second one $z_1$ outputs the probability that instances belong to a class labeled $1$. In fewer words, for $z_0$ the target class is labeled $0$, and for $z_1$ the target class is labeled $1$. To activate $z_0$ and $z_1$ with softmax we compute:

$$Softmax(z_0 ) = \frac{e^{z_0}}{e^{z_0} + e^{z_1}} \\$$
$$Softmax(z_1 ) = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$

*Softmax is applied to each neuron in the output layer. In addition to mapping all neurons from $ℝ$ to the range $[0,1]$, it makes their values add up to $1$.*

Now we can remediate the seeming lack of an apparent link between $sigmoid$ and $softmax$. We will do this by simply rewriting $sigmoid$:
### $$\sigma(z) = \frac{1}{1+e^{-z}} = \frac{1}{1+\frac{1}{e^{z}}} = \frac{1}{\frac{e^{z}+1}{e^{z}}} = \frac{e^{z}}{1+e^{z}} = \frac{e^{z}}{e^0+e^{z}}$$
*$sigmoid$, is actually $softmax$ with two classes.*

It is more common to see the first mentioned version of $sigmoid$ than it is to see the second one. This is because the latter version is more expensive computationally. Its advantage here remains in helping us understand $softmax$.

With only two neurons in the output layer, and given the fact that $softmax$ makes all output neurons sum up to $1$: we always know that $Softmax(z_0)$ is going to be equal to $1-Softmax(z_1)$. Hence for binary classification, it makes sense to consider $z_0$ equal to $0$, and to only compute the activation of $z_1$ using $sigmoid$.

The following code implements $softmax$ and tests it with an example of three output neurons. Then it compares our result with the result of Pytorch’s [softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax).

```python
import torch
import torch.nn.functional as F

#----------- Implementing the math -----------#

def softmax(z):
  return z.exp() / z.exp().sum(axis=1, keepdim=True) 
  # keepdim=True tells sum() that we want its output to have the same dimension as z 

zs = torch.tensor([[2., 3., 1.]]) # Three output neurons

sm = softmax(zs)

#----------- Using Pytorch -----------#

torch_sm = F.softmax(zs, dim=1)

#----------- Comparing outputs -----------#

print(f"Pytorch Softmax: \n{torch_sm} \nOur Softmax: \n{sm}")
'''
Out:
Pytorch Softmax: 
tensor([[0.2447, 0.6652, 0.0900]]) 
Our Softmax: 
tensor([[0.2447, 0.6652, 0.0900]])
'''
```
We differentiate $softmax$ activations with respect to each neuron. Keeping the same example of an output layer with two neurons, we get four $softmax$ differentiations:

{% include dismantling_imgs/sm/sms@2x.png %}

*The four values constitute the Jacobian matrix of $softmax$.*

Regardless of the number of output neurons, there are only two formulas for softmax differentiation. The first formula is applied when we differentiate the softmax of a neuron with respect to itself (top left and bottom right differentiations in the Jacobian). The second formula is applied when we differentiate the softmax of a neuron with respect to some other neuron (top right and bottom left differentiations in the Jacobian).

To understand the steps involved in the differentiation of softmax, we need to recall one more differentiation rule:

{% include dismantling_imgs/sm/division-rule@2x.png %}

*The division rule*

The following differentiations contain detailed steps. And although they might seem intimidating by the fact that they look dense, I assure you that they are much easier than they look, and I encourage you to redo them on paper.

{% include dismantling_imgs/sm/p_diffs_sm.png %}

The implementation of the $softmax$ differentiation requires us to iterate through the list of neurons and differentiate with respect to each neuron. Hence two loops are involved. Keep in mind that the purpose of these implementations is not to be efficient, but rather to explicitly translate the math and arrive at the same results achieved by the built-in methods of Pytorch.

```python
import torch
import torch.nn.functional as F

#----------- Implementing the math -----------#
def softmax(z):
  return z.exp() / z.exp().sum(axis=1, keepdim=True)

def softmax_prime(z):
  sm = softmax(z).squeeze()
  sm_size = sm.shape[0]
  sm_ps = []
  for i, sm_i in enumerate(sm):
    for j, sm_j in enumerate(sm):
      # First case: i and j are equal:
      if(i==j):
        # Differentiating the softmax of a neuron w.r.t to itself
        sm_p = sm_i * (1 - sm_i)
        sm_ps.append(sm_p)
      # Second case: i and j are not equal:
      else:
        # Differentiating the softmax of a neuron w.r.t to another neuron
        sm_p = -sm_i * sm_j
        sm_ps.append(sm_p)
  sm_ps = torch.tensor(sm_ps).view(sm_size, sm_size)
  return sm_ps

        
z = torch.tensor([[4., 2.]], requires_grad=True)
sm_p = softmax_prime(z)

#----------- Using Pytorch autograd -----------#

torch_sm = F.softmax(z, dim=1)

# to extract the first row in the jacobian matrix, use [[1., 0]] 
# retain_graph=True because we re-use backward() for the second row
torch_sm.backward(torch.tensor([[1.,0.]]), retain_graph=True) 
r1 = z.grad
z.grad = torch.zeros_like(z) 
# to extract the second row in the jacobian matrix, use [[0., 1.]] 
torch_sm.backward(torch.tensor([[0.,1.]])) 
r2 = z.grad
torch_sm_p = torch.cat((r1,r2))

#----------- Comparing outputs -----------#

print(f"Pytorch Softmax': \n{torch_sm_p} \nOur Softmax': \n{sm_p}")
'''
Out:
Pytorch Softmax': 
tensor([[ 0.1050, -0.1050],
        [-0.1050,  0.1050]]) 
Our Softmax': 
tensor([[ 0.1050, -0.1050],
        [-0.1050,  0.1050]])
'''
```

## Cross-Entropy Loss
In the sequence of operations involved in a neural network, softmax is generally followed by the cross-entropy loss. In fact, the two functions are so closely connected that in Pytorch the method [cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy) combines both functions in one.

I remember my first impression when I saw the formula for the cross-entropy loss. It was close to admiring hieroglyphs. After deciphering it, I hope you will share my awe towards how simple ideas can sometimes have the most complex representations.

{% include dismantling_imgs/cross-entropy@2x.png %}

The variables involved in calculating the cross-entropy loss are $p$, $y$, $m$, and $K$. Both $i$ and $k$ are used as counters to iterate from $1$ to $m$ and from $1$ to $K$ respectively.

- $Z$: is an array where each row represents the output neurons of one instance. m: is the number of instances.

- $K$: is the number of classes.

- $p$: is the probability of the neural network that instance $i$ belongs to class $k$. This is the same probability computed from softmax.

- $y$: is the label of instance $i$. It is either $1$ or $0$ depending on whether $y$ belongs to class $k$ or not.

- $log$: is the natural logarithm.

Let’s say we are performing a multi-class classification task where the number of possible classes is three $(K=3)$. Each instance can only belong to one class. Therefore each instance is assigned to a vector of labels with two zeros and a one. For example $y=[0,0,1]$ means that the instance of $y$ belongs to class $2$. Similarly, $y=[1,0,0]$ means that the instance of $y$ belongs to class $0$. The index of the $1$ refers to the class to which the instance belongs. We say that the labels are *one-hot encoded*.

Now let’s take two instances $(m=2)$. We calculate their $z$ values and we find: $Z = [[0.1, 0.4, 0.2], [0.3, 0.9, 0.6]]$. Then we calculate their softmax probabilities and find: $Activations = [[0.29, 0.39, 0.32], [0.24, 0.44, 0.32]]$. We know that the first instance belongs to class $2$, and the second instance belongs to class $0$, because: $y = [[0,0,1],[1,0,0]]$.

To calculate cross-entropy:

1. We take the $log$ of the softmax activations: $log(activations) = [[-1.24, -0.94, -1.14], [-1.43, -0.83, -1.13]]$.
2. We multiply by $-1$ to get the negative $log$: $-log(activations) = [[1.24, 0.94, 1.14], [1.43, 0.83, 1.13]]$.
3. Multiplying $-log(activations)$ by $y$ gives: $[[0., 0., 1.14], [1.43, 0., 0.]]$.
4. The sum over all classes gives: $[[0.+0.+1.14], [1.43+0.+0.]] = [[1.14], [1.43]]$
5. The sum over all instances gives: $[1.14+1.43] = [2.57]$
6. The division by the number of instances gives: $[2.57 / 2] = [1.285]$

**Observations:**

- Steps 3 and 4 are equivalent to simply retrieving the negative $log$ of the target class.

- Steps 5 and 6 are equivalent to calculating the mean.

- The loss is equal to $1.14$ when the neural network predicted that the instance belongs to the target class with a probability of $0.32$.

- The loss is equal to $1.43$ when the neural network predicted that the instance belongs to the target class with a probability of $0.24$.

- We can see that in both instances the network failed to give the highest probability to the correct class. But compared to the first instance, the network was more confident about the second instance not belonging to the correct class. Consequently, it was penalized with a higher loss of $1.43$.

We combine the above steps and observations in our implementation of cross-entropy. As usual, we will also go through the Pytorch equivalent method, before comparing both outputs.

```python
import torch
import torch.nn.functional as F

#----------- Implementing the math -----------#
def cross_entropy(activations, labels):
  return - torch.log(activations[range(labels.shape[0]), labels]).mean()

zs = torch.tensor([[0.1, 0.4, 0.2], [0.3, 0.9, 0.6]]) # The values of 3 output neurons for 2 instances
activations = softmax(zs) # = [[0.2894, 0.3907, 0.3199],[0.2397, 0.4368, 0.3236]]
y = torch.tensor([2,0]) # equivalent to [[0,0,1],[1,0,0]]
ce = cross_entropy(activations, y)

#----------- Using Pytorch autograd -----------#
torch_ce = F.cross_entropy(zs, y)

#----------- Comparing outputs -----------#
print(f"Pytorch cross-entropy: {torch_ce} \nOur cross-entropy: {ce}")
'''
Out:
Pytorch cross-entropy: 1.28411 
Our cross-entropy: 1.28411
'''
```
**Note:** 

Instead of storing the one-hot encoding of the labels, we simply store the index of the $1$. For example, the previous $y$ becomes $[2,0]$. Notice, at index $0$ the value of $y$ is $2$, and at index $1$ the value of $y$ is $0$. Using the indices of $y$ and their values, we can directly retrieve the negative logs for the target classes. This is done by accessing $-log(activations)$ at row $0$ column $2$, and at row $1$ column $0$. This allows us to avoid the wasteful multiplications and additions of zeros in steps 3 and 4. This trick is called integer array indexing and is explained by Jeremy Howard in his Deep Learning From The Foundations [lecture 9](https://www.youtube.com/watch?v=AcA8HAYh7IE&list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&index=2) at 34:57.

<br/><br/>
***
<br/><br/>
If going forward through the layers of a neural network can be seen as its journey to acquire some kind of knowledge, then we have arrived at the place where that knowledge can be found. Using the differentiation of a loss function can inform the neural network of how much it erred on each instance. Taking this error backwards, a neural network can adjust itself.

{% include dismantling_imgs/ce_p@2x.png %}

*Cross-entropy differentiation.*

We go through the differentiation steps of cross-entropy after we recall a couple differentiation rules:

{% include dismantling_imgs/ce_rules@2x.png %}

*Recall these two differentiation rules. Also recall that $ln$ is the same as $log$ based $e$. The base $e$ is assumed throughout the article.*

{% include dismantling_imgs/ce_p_steps@2x.png %}

*Cross-entropy differentiation steps.*

We will not be able to evaluate the following implementation with the output of Pytorch Autograd just yet. The reason goes back to Pytorch’s [cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy) combining softmax with cross-entropy. Consequently, using [backward](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward) would also involve the differentiation of softmax in the chain rule. We discuss and implement this in the next section, **Backpropagation**. For now, here is our implementation of cross-entropy’ :

```python
import torch 

zs = torch.tensor([[0.1, 0.4, 0.2], [0.3, 0.9, 0.6]]) # The values of 3 output neurons for 2 instances
activations = softmax(zs) # = [[0.2894, 0.3907, 0.3199],[0.2397, 0.4368, 0.3236]]
y = torch.tensor([2,0]) # equivalent to [[0,0,1],[1,0,0]]

#----------- Implementing the math -----------#
def crossentropy_prime(activations, labels):
  n = labels.shape[0]
  activs = torch.zeros_like(activations)
  activs[range(n), labels] = -1 / activations[range(n), labels] # integer array indexing
  return activs 

c_p = crossentropy_prime(activations, y)

#----------- Printing Output -----------#
print(f"Cross-entropy differentiation: \n{c_p}\n")
'''
Out:
Cross-entropy differentiation: 
tensor([[ 0.0000,  0.0000, -3.1262],
        [-4.1720,  0.0000,  0.0000]])
'''
```
<br/><br/>
## Backpropagation
With every function we discussed, we made one step forward in the layers of a neural network, and we also made the equivalent step backward using the differentiation of the functions. Since neural networks move all the way forward before retracing their steps all the way backward, we need to discuss how to link our functions.

### Forward

Going all the way forward, a neural network with one hidden layer starts by feeding input to a linear function, then feeds its output to a non-linear function, then feeds its output to a loss function. The following is an example with an instance $x$, its corresponding label $y$, three linear neurons $z$ (each neuron computed using its three weights $w$ and a bias $b$), followed by a softmax activation layer, and a cross-entropy loss.

```python
y = torch.tensor([2])
x = torch.tensor([[0.9, 0.5, 0.3]])
w = torch.tensor([[0.2, 0.1, 0.4], [0.5, 0.6, 0.1], [0.1, 0.7, 0.2]], requires_grad=True)
b = torch.tensor([[0.1, 0.2, 0.1]], requires_grad=True)

#----------- Using our functions -----------#

z = x @ w.T + b # = [[0.4500, 0.9800, 0.6000]]

sm = softmax(z) # = [[0.2590, 0.4401, 0.3009]]

ce = cross_entropy(sm, y) # = 1.2009

#----------- Equivalent Pytorch -----------#

t_ce = F.cross_entropy(z, y) # = 1.2009
```

<br/><br/>

### Backward

Going all the way backward, the same neural network starts by taking the same input given to the loss function, and feeds it instead to the derivative of that loss function. The output of the derivative of the loss function is the error, what we called the acquired knowledge. To adjust its parameters, the neural network has to carry this error another step backwards to the non-linear layer, and from there another step backwards to the linear layer.

The next step backward is not as simple as feeding the error to the derivative of the non-linear function. We need to use the chain rule (which we previously recalled in the differentiation of sigmoid), and we also need to pay attention to the input we should give to each derivative.

***Key rules for feedforward and backpropagation:***
- Functions and their derivatives take the same input.

- Functions send their output forward to be the input of the next function.

- Derivatives send their output backward to multiply the output of the previous derivative.

```python
#----------- Using our differentiations -----------#

ce_p = crossentropy_prime(sm, y) # = [[ 0.0000,  0.0000, -3.3230]]

sm_p = softmax_prime(z) # = [[ 0.1919, -0.1140, -0.0779],
                          #  [-0.1140,  0.2464, -0.1324],
                          #  [-0.0779, -0.1324,  0.2104]])

z_p_w = torch.stack(([x]*3)).squeeze()  # Recall: z' w.r.t the weights is equal to x
z_p_b = torch.ones_like(b)              # Recall: z' w.r.t the biases is equal to 1

# Backwards from cross-entropy to softmax
ce_sm = (ce_p @ sm_p.T) 

# Backwards from softmax to z
our_w_grad = ce_sm.T * z_p_w 
our_b_grad = ce_sm * z_p_b 

#----------- Using Pytorch Autograd -----------#
t_ce.backward()
t_w_grad = w.grad
t_b_grad = b.grad

#----------- Comparing Outputs -----------#
print(f"Pytorch w_grad: \n{t_w_grad} \nPytorch b_grad: \n{t_b_grad}")
print(f"Math w_grad: \n{our_w_grad} \nMath b_grad: \n{our_b_grad}")
'''
Out:
Pytorch w_grad: 
tensor([[ 0.2331,  0.1295,  0.0777],
        [ 0.3960,  0.2200,  0.1320],
        [-0.6292, -0.3495, -0.2097]]) 
Pytorch b_grad: 
tensor([[ 0.2590,  0.4401, -0.6991]])
Math w_grad: 
tensor([[ 0.2331,  0.1295,  0.0777],
        [ 0.3960,  0.2200,  0.1320],
        [-0.6292, -0.3495, -0.2097]]) 
Math b_grad: 
tensor([[ 0.2590,  0.4401, -0.6991]])
'''
```

*May your machine’s output always be in accordance with your math.*
<br/><br/>

## Conclusion
My impression is that a lot of people with backgrounds from different disciplines are curious and enthusiastic towards Machine Learning. Unfortunately, there is a justifiable trend towards acquiring the know-how while trying to keep away from the intimidating math. I consider this unfortunate because I believe many people are actually eager to deepen their understanding; If only they could find more resources that appeal to the fact that they come from different backgrounds and might need a little reminder and a little encouragement here and there.

This article contained my attempt towards writing reader-friendly math. By which I mean math that reminds the reader of the rules required to follow along. And by which I also mean math with equations that avoid skipping so many steps and making us ponder what happened between one line and the next. Because sometimes we really need someone to take our hand and walk with us through the fields of unfamiliar concepts. My sincere hope that I was able to reach your hand.

## References
M. Amine, [The Inner Workings of Neural Networks, My Colab Notebooks](https://drive.google.com/drive/folders/1EkhYkKE74Kz2rYCvGM1BMTJ5ih7Q-toA) (2020).

M. Amine, [The Inner Workings of Neural Networks, My Gists](https://gist.github.com/Mehdi-Amine), (2020).

A. Géron, [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/), (2019).

S. Gugger, [A simple neural net in numpy](https://sgugger.github.io/a-simple-neural-net-in-numpy.html), (2018).

J. Howard, [Fast.ai: Deep Learning from the Foundations Lesson 9](https://www.youtube.com/watch?v=AcA8HAYh7IE&t=2135s), (2019).

Pytorch, [Pytorch documentation](https://pytorch.org/docs/stable/index.html).