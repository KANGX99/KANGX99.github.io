---
author: "Kangx"
title: "How does LSTM avoid gradient explosion or vanish"
date: "2024-06-11"
tags: ["LSTM","RNN","gradient vanish/explosion"]
ShowToc: true
ShowBreadCrumbs: false
---

## How does LSTM avoid gradient explosion or vanish?

<center>
    <img style="border-radius: 0.3125em;" 
    src=".\RNN.png">
    <br>
    <div style="color:orange; display: inline-block;
    color: #999;
    padding: 2px;">RNN architecture</div>
</center>

### Why can't RNN capture long-term dependencies?

$$
h^{(t)} = tanh(Ux^{(t)}+Wh^{(t-1)}+b)\\
o^{(t)} = Vh^{t}+c\\
\hat{y}^{(t)} = softmax(o^{(t)})
$$



We can explain why RNN can't capture long-term dependency from two angles:

1. **From the angle of $\frac{\partial h^{(t)}}{\partial h^{(1)}}$:**

    ```
    \frac{\partial h^{(t)}}{\partial h^{(1)}} = \prod_{j=2}^{t} \frac{\partial h^{(j)}}{\partial h^{(j-1)}} = \prod_{j=2}^{t} tanh' \cdot W
    ```

    This derivative essentially tells us how much our hidden state at time $t$ will change when we change the hidden state at time $1$ by a little bit. According to the above math, if the gradient vanishes it means the earlier hidden states have no real effect on the later hidden states, meaning no long term dependencies are learned!

2. **From the angle of $\frac{\partial L}{\partial r_t}$:**

    ```
    \frac{\partial L}{\partial r_t} = diag(1-tanh^2r_t) \cdot U^T \cdot \frac{\partial L}{\partial r_{t+1}} + diag(1-tanh^2r_t) \cdot V^T \cdot \frac{\partial L}{\partial z_t}
    ```

    ```
    r_t = U \cdot h_{t-1} + W \cdot x_t + b
    ```

    From the above formulas, we can observe that when $t$ is very small and the dominant eigenvalue of the matrix $U$ is less than 1, $\frac{\partial L}{\partial r_t}$ will be close to 0. As such, if we change $r_t$ by a little bit, loss $L$ will basically not change. If our task relies more on historical information, this is obviously unreasonable.


<br/>

### LSTM Structure

<br/>

The forward propagation process of LSTM is as follows:

* **Forget Gate:** 
    ```
    f_t = \sigma(W_f[h_{t-1}, x_t]) 
    ```
* **Input Gate:**
    ```
    i_t = \sigma(W_i[h_{t-1}, x_t])
    ```
* **Output Gate:**
    ```
    o_t = \sigma(W_o[h_{t-1}, x_t])
    ```
* **Candidate Memory Cell:**
    ```
    \widetilde{C}_t = tanh(W_C[h_{t-1}, x_t])
    ```
* **Memory Cell Update:**
    ```
    C_t = f_t C_{t-1} + i_t \widetilde{C}_t
    ```
* **Hidden State Update:**
    ```
    h_t = o_t tanh(C_t)
    ```

<br/>

We can calculate $\frac{\partial h^{(j)}}{\partial h^{(j-1)}}$ as above, but from Eqn.(12) we can see that analyzing $C_t$ is equivalent to analyzing $h_t$, and calculating $\frac{\partial C^{(t)}}{\partial C^{(t-1)}}$ is simpler.

```
\begin{aligned}
\frac{\partial C_t}{\partial C_{t-1}} & = C_{t-1} \sigma'(.) W_f * o_{t-1} tanh'(C_{t-1}) \\
& + \widetilde{C}_t \sigma'(.) W_i * o_{t-1} tanh'(C_{t-1}) \\
& + i_t tanh'(.) W_C * o_{t-1} tanh'(C_{t-1}) \\
& + f_t
\end{aligned}
```

<br/>

Now if we want to backpropagate back $k$ time steps, we simply multiply terms in the form of the one above 
$k$ times.The terms here, $\frac{\partial C_t}{\partial C_{t-1}}$, at any time step can take on either values that are greater than 1 or values in the range 
[0,1].

If we start to converge to zero, we can always set the values of $f_t$ (and other gate values) to be higher in order to bring the value of $\frac{\partial C_t}{\partial C_{t-1}}$ closer to 1, thus preventing
