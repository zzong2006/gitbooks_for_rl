---
layout: post
title: 'Reinforcement learning: Temporal-Difference Learning'
date: '2020-10-23 22:55:30 +0900'
categories: reinforcement_learning
series: 4
---

# index

* TD = DP ideas + MC ideas
  * MC ideas: 환경의 dynamics 모델 없이 raw experience\(sample\)에서 학습 가능
  * DP ideas: bootstrap함 \(다른 estimates를 이용해 estimate 업데이트 가능\)

## TD Prediction

* MC와 TD는 둘 다 경험을 통해 $v\_\pi$의 추정값 $V$를 찾는다.

### Constant-$\alpha$ MC

* MC 방법\(특히 every-visit\)은 $V\(S\_t\)$를 추정하기 위해 **실제** return 값 $G\_t$를 알아야한다.
  * $$
    V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left[G_{t}-V\left(S_{t}\right)\right]
    $$

    * $\alpha$는 constant step-size \($0 \lt \alpha \le 1$\) 
  * episode가 완료될 때 $G\_t$를 거꾸로 찾아가므로, episode가 완료되기까지 위의 update는 불가능하다.

### TD\(0\): one-step TD

* TD는 MC와 다르게 next time step까지만 기다린다.
  * 즉, $S_t$에서 $S_{t+1}$로 넘어간 순간 바로 $R_{t +1}$와 $V\(S_{t+1}\)$의 **추정**값을 이용해 update를 시도한다.
  * $$
    V\left(S_{t}\right) \leftarrow         V\left(S_{t}\right)+\alpha\left[R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right]
    $$
  * ![image-20201023163536850](https://i.loli.net/2020/10/23/Z2ctdA6YiTEWuoH.png)
* 식에서 확인할 수 있듯이, MC의 update는 $G_t$이고, TD의 update는 $R_{t+1}+\gamma V\(S\_{t+1}\)$ 이다.
* ![image-20201023162844126](https://i.loli.net/2020/10/23/7dTWvMGcRA1NnLB.png)

#### Bootstrap of TD

* TD\(0\)는 기존의 추정에 기반하여 update를 진행하므로, bootstrapping 방법을 이용한다고 말할 수 있다\(like DP\).
* $$
  \begin{aligned}
  v_{\pi}(s) & \doteq \mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right] \\
  &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s\right] \\
  &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) \mid S_{t}=s\right]
  \end{aligned}
  $$

  * 위 식에서 MC는 $\mathbb{E}_{\pi}\left\[G_{t} \mid S\_{t}=s\right\]$ 를 target으로하여 추정을 진행한다.
    * 추정인 이유는 $G\_t$값을 정확히 알 수 없기 때문이다.
  * DP는 $\mathbb{E}_{\pi}\left\[R_{t+1}+\gamma v_{\pi}\left\(S_{t+1}\right\) \mid S\_{t}=s\right\]$을 target으로하여 추정을 진행한다.
    * 추정인 이유는 $v_\pi\(S_{t+1}\)$값을 정확히 알 수 없어서 $V\left\(S\_{t+1}\right\)$를 사용하기 때문이다.
  * 그리고 TD는 위 두 이유로 추정을 진행한다.
    * TD는 expected value를 sampling하고, 이 값을 이용해 $V$를 estimate하여 실제 $v\_\pi$를 찾는데 사용한다.
      * 이런 방법을 sample updates라고 한다.
    * 즉, DP의 bootstrapping과 MC의 sampling을 합친 방법과 같다.

#### TD error

* TD error란, $S_t$의 기존 추정값과 더 나은 추정값 $R_{t+1}+\gamma V\left\(S\_{t+1}\right\)$ 의 차이를 말한다.
  * $$
    \delta_{t} \doteq R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)
    $$
* TD error는 그 시간대에서만 유효하고, 이후에는 더 이상 유효하지 않다.
  * 즉, $V\(S\_t\)$의 에러 $\delta\_t$는 $t+1$에서만 유효하다.
* 또한, 한 에피소드 내에서 $V$가 업데이트 되지않고 쭉 유지된다면 MC와 같을 것이다. 즉, MC error는 TD error들의 합으로 표현될 수 있다.
  * $$
    \begin{aligned}
    G_{t}-V\left(S_{t}\right) &=R_{t+1}+\gamma G_{t+1}-V\left(S_{t}\right)+\gamma V\left(S_{t+1}\right)-\gamma V\left(S_{t+1}\right) \\
    &=\delta_{t}+\gamma\left(G_{t+1}-V\left(S_{t+1}\right)\right) \\
    &=\delta_{t}+\gamma \delta_{t+1}+\gamma^{2}\left(G_{t+2}-V\left(S_{t+2}\right)\right) \\
    &=\delta_{t}+\gamma \delta_{t+1}+\gamma^{2} \delta_{t+2}+\cdots+\gamma^{T-t-1} \delta_{T-1}+\gamma^{T-t}\left(G_{T}-V\left(S_{T}\right)\right) \\
    &=\delta_{t}+\gamma \delta_{t+1}+\gamma^{2} \delta_{t+2}+\cdots+\gamma^{T-t-1} \delta_{T-1}+\gamma^{T-t}(0-0) \\
    &=\sum_{k=t}^{T-1} \gamma^{k-t} \delta_{k}
    \end{aligned}
    $$

## Advantages of TD Prediction Methods

* TD 방법 vs DP 방법
  * 환경에 대한 모델이 필요하지 않고, 다름 상태에 대한 확률 분포가 요구되지 않는다.
* TD 방법 vs MC 방법
  * 한 episode가 끝날때 까지 기다리지 않아도 괜찮다.
  * 어떤 task는 굉장히 긴 episode가 있기 때문에 학습이 매우 느려질 수 있다.
* 그리고 TD 방식은 고정된 policy $\pi$에 대해서 TD\(0\)을 통해 $v\_\pi$가 수렴하는 것이 보장되어 있다.
* 아직까지 수학적으로 증명되진 않았지만, TD 방법이 실제로 constant-$\alpha$ MC 방법보다 빠르게 수렴하는 것으로 알려져있다.
  * 하지만 $\alpha$값이 적절히 잡혀야함
  * ![image-20201023164827689](https://i.loli.net/2020/10/23/F52MmP6peYJVUzW.png)

## Optimality of TD\(0\)

* batch updating: 일괄\(batch\) 학습 데이터를 통해 updates가 진행되는 경우를 의미
  * TD\(0\) 또는 constant-$\alpha$ MC는 sample을 통해 추정값의 update를 진행한다.
* Batch MC 방법은 언제나 학습 데이터에 대한 MSE\(mean-squared error\)를 최소화하려고 한다.
* 반대로, TD\(0\)는 언제나 Markov process의 maximum-likelihood 모델을 추정하려한다. 

이제 TD prediction 방법을 control 문제에 적용해보자.

GPI 패턴을 활용하고 TD 방법을 evaluation 또는 prediction 부분에 사용한다.

MC 방법과 마찬가지로, exploration과 exploitation의 trade off가 존재하고, on-policy와 off-policy의 두 main classes가 존재한다.

## SARSA: On-policy TD Control

* SARSA는 state값 말고, action 값을 이용하여 policy를 improve하는 방법이다.
  * 즉, transitions from state–action pair to state–action pair를 고려해야한다.  
  * 이전의 TD\(0\)에서 state 대신 action 값을 이용한것이라 생각하면 된다.
* ![image-20201023172332416](https://i.loli.net/2020/10/23/WZvHVdpfSL3CAPI.png)
* $$
  Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma Q\left(S_{t+1}, A_{t+1}\right)-Q\left(S_{t}, A_{t}\right)\right]
  $$

  * ![image-20201023172646120](https://i.loli.net/2020/10/23/u2LqvWVTtze8Ujr.png)
  * 여기서 $S_{t+1}$가 terminal이면,  $Q\left\(S_{t+1}, A\_{t+1}\right\)$ 은 0으로 정한다.
* 매 학습마다 $\left\(S_{t}, A_{t}, R_{t+1}, S_{t+1}, A\_{t+1}\right\)$ 를 사용하기 때문에 SARSA라 불린다.
* ![image-20201023173337312](https://i.loli.net/2020/10/23/UbZQXC3GcDFylAs.png)

## Q-learning: Off-policy TD Control

* $$
  Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma \max _{a} Q\left(S_{t+1}, a\right)-Q\left(S_{t}, A_{t}\right)\right]
  $$

  * $Q$에서 유도되는 policy에 상관없이 $\max _{a} Q\left\(S_{t+1}, a\right\)$ 를 통해서 최적의 action-value function를 통해 $Q$를 update한다.
  * 즉, greedy policy가 하나 있고, $\varepsilon$-greedy와 같은 또 다른 policy가 존재하므로 Q-learning을 off-policy 라 부를 수 있다.
* ![image-20201023175619321](https://i.loli.net/2020/10/23/a6LDEiPeUukvNgX.png)
* ![image-20201023180629700](https://i.loli.net/2020/10/23/Bp152rmX3Rlsdi9.png)

## Expected SARSA

* SARSA에서 기댓값을 사용한 SARSA 버전
  * $$
    \begin{aligned}
    Q\left(S_{t}, A_{t}\right) & \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma \mathbb{E}_{\pi}\left[Q\left(S_{t+1}, A_{t+1}\right) \mid S_{t+1}\right]-Q\left(S_{t}, A_{t}\right)\right] \\
    & \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma \sum_{a} \pi\left(a \mid S_{t+1}\right) Q\left(S_{t+1}, a\right)-Q\left(S_{t}, A_{t}\right)\right]
    \end{aligned}
    $$
  * 즉, $Q\left\(S_{t+1}, A_{t+1}\right\)$ 대신 $\mathbb{E}_{\pi}\left\[Q\left\(S_{t+1}, A_{t+1}\right\) \mid S_{t+1}\right\]$ 를 사용함
* Expected SARSA는 SARSA보다 계산이 복잡하지만, $A\_{t+1}$를 무작위로 선택할 때 발생하는 분산을 없애준다.
* 일반적으로 Expected SARSA는 action을 결정하기 위해서 $\pi$와는 다른 정책\(Expected value\)을 사용할 것이고, 이 경우에는 off-policy algorithm이 된다.  

## Maximization Bias and Double Learning

* Q-Learning과 SARSA 둘 다 최대라는 개념을 활용하는데, 이럴 경우 maximization bias 문제가 발생할 수 있다.
  * Maximization bias: $Q\(s,a\)$의 추정값이 일부는 양수고 일부는 음수를 가져서 최대값을 활용하는 action 선택에 영향을 미치는 것
* 쉽게 예를 들어보자. 아래와 같은 MDP에서 right는 0의 보상 그리고 left는 평균 -0.1과 분산 1.0의 정규 분포를 따르는 보상을 준다.
  * 결과적으로 left 에 대한 이득의 기댓값은 -0.1이고 right를 선택하는 것이 옳다.
  * 그러나 분산에 의해 maximization bias가 발생하여 Q-learning은 left를 선택할 것이다\(보상이 종종 양수가 나오므로 right의 0보다 크기 때문에 최대값으로 right 선택\).
  * ![image-20201023202144253](https://i.loli.net/2020/10/23/oFspJj3d94fOmLq.png)
* maximization bias가 q-learning같은 학습에 영향을 미치는 원인은 maximizing action을 결정하는 것과 그 action의 값을 추정하는데 **동일한 samples**을 사용하기 때문이다.

### Solution: Double Learning

* 두 Q 값 $Q\_1\(a\)$와 $Q\_2\(a\)$가 존재한다고 해보자. 
  * 각각은 모든 $a \in \mathcal{A}$에 대해서 $q\(a\)$값을 찾으려\(추정하려\)한다.
  * 그리고 한 $Q$는 maximizing action 결정에, 다른 $Q$는 결정된 action의 value 업데이트에 쓰이게 한다.
    * $A^{_}=\arg \max {a} Q{1}\(a\)$ 그리고, $Q\_{2}\left\(A^{_}\right\)=Q_{2}\left\(\arg \max_ {a} Q\_{1}\(a\)\right\)$
    * 즉, $\mathbb{E}\left\[Q\_{2}\left\(A^{_}\right\)\right\]=q\left\(A^{_}\right\)$를 목적으로 학습한다.
  * 이렇게 하면 한쪽 $Q$에 사용된 sample의 목적\(action 선택\)이 다른 한쪽 $Q$에 사용된 sample의 목적\(value 업데이트\)와 달라서 unbias 하게 된다.
  * 종종 이 둘의 역할을 바꿀 수 있다:  $Q_{1}\left\(\arg \max_ {a} Q\_{2}\(a\)\right\)$
* ![image-20201023203043286](https://i.loli.net/2020/10/23/dmeBwntWSg1lkqE.png)
  * behavior policy에는 두 $Q$ 값을 동시에 사용할 수 있다\(더하거나 평균내서\)
  * 그리고 동전을 던지듯이 $0.5$의 확률로 $Q$의 역할을 교체하면서 수행하면 된다.

