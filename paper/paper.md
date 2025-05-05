# Flexible Robust Beamforming for Multibeam

# Satellite Downlink using Reinforcement Learning

### Alea Schroder, Steffen Gracla, Maik R ̈ oper, Dirk W ̈ ubben, Carsten Bockelmann, Armin Dekorsy ̈

```
Dept. of Communications Engineering, University of Bremen, Bremen, Germany
Email:{schroeder, gracla, roeper, wuebben, bockelmann, dekorsy}@ant.uni-bremen.de
```
```
Abstract
Low Earth Orbit (LEO) satellite-to-handheld connections herald a new era in satellite communications. Space-Division Multiple
Access (SDMA) precoding is a method that mitigates interference among satellite beams, boosting spectral efficiency. While optimal
SDMAprecoding solutions have been proposed for ideal channel knowledge in various scenarios, addressing robust precoding with
imperfect channel information has primarily been limited to simplified models. However, these models might not capture the
complexity of LEO satellite applications. We use the Soft Actor-Critic (SAC) deep Reinforcement Learning (RL) method to learn
robust precoding strategies without the need for explicit insights into the system conditions and imperfections. Our results show
flexibility to adapt to arbitrary system configurations while performing strongly in terms of achievable rate and robustness to
disruptive influences compared to analytical benchmark precoders.
```
```
Index Terms
6G, Multi-user beamforming, 3D networks, Low Earth Orbit (LEO), Satellite Communications, Machine Learning (ML), deep
Reinforcement Learning (RL)
```
#### I. INTRODUCTION

The upcoming sixth generation standard of mobile communications will seek to extend our classic terrestrial networks to
so-called 3D networks by integrating communication satellites and Unmanned Aerial Vehicles (UAVs) [1]. This additional degree
of freedom is expected to boost continuous global coverage, improve balancing traffic demand surges, and increase outage
protection [2], [3]. In contrast to higher satellite orbits, the Low Earth Orbit (LEO) offers relatively low latency and path loss,
and reduced deployment cost. Therefore,LEOsatellites are targeted as a key component of future Non Terrestrial Networks
(NTN) [1]. The satellites’ downlink transmit power can be steered by Space-Division Multiple Access (SDMA) precoding. The
precoder optimizes each users’ signal power while mitigating inter-user interference, achieving better spectral efficiency [4].
ConventionalSDMAprecoding techniques, however, are challenged by errors in position estimation caused by outdated Channel
State Information at Transmitter (CSIT), which can degrade the performance quickly [5]. Under real circumstances, a variety of
perturbing effects may further influence the transmission quality. They are induced by, for example, the high relative velocity
ofLEOsatellites and atmospheric influences.
In order to design a robust precoder, the authors in [5] maximize the achievable rate, taking imperfect position knowledge
into account, and propose a low complexity algorithm using supervised learning. In [6], an analytical robust precoder is derived
that utilizes the second order statistics of the channel to maximize the mean Signal-to-Leakage-and-Noise Ratio (SLNR). [7]
investigate Rate-Splitting Multiple Access (RSMA) to deal with errors in position measurements, by splitting the user messages
into individually precoded common and private parts. In our previous paper [8], we take a first look at using deep Reinforcement
Learning (RL) for the purpose of beamforming in the presence of imperfectCSIT. Deep Learning (DL), as a sub-set of Machine
Learning (ML), infers and iteratively tunes the parameters of a Neural Network (NN) based on a data set such that theNN
approximately maximizes an objective. These data-drivenDLapproaches have achieved noteworthy performance across most
domains in the past decade, and their application in communication networks has been under intense study, e.g., [9].MLis of
particular interest for robust algorithms since input data with uncertainty are often seen as desirable and positive during the
learning process [10, Chp. 7.5]: to prevent over-fitting on the available data, practitioners will frequently add noise on their
data as a regularizing measure. In [8], we assume a multi-satellite downlink scenario and useRLto train aNNthat takes as
input a channel matrix estimate to output a precoding matrix that approximately maximizes the achievable sum rate. We use
the Soft Actor-Critic (SAC)RLalgorithm [11] that allows for continuous-valued output and maintains a measure of uncertainty
about its decisions. This measure of uncertainty is used to guide the learning process to be sample efficient. We demonstrate
the feasibility of this approach and its strong robustness to disruptive influences comparing to the common Minimum Mean
Squared Error (MMSE) precoding approach.
In this paper we move on to a single satellite downlink scenario that is more challenging due to covering much larger
user distances. We compare our performance to the popularMMSEprecoder, as well as the above mentioned analytical robust
precoder [6], both of which strongly favor this spatially decoupled scenario. In Section II and Section III, we first discuss

This work was partly funded by the German Ministry of Education and Research (BMBF) under grant 16KISK016 (Open6GHub) and 16KIS
(MOMENTUM) and the European Space Agency (ESA) under contract number 4000139559/22/UK/AL (AIComS).


the satellite downlink model and the two benchmark precoders. Afterwards, we outline the learning approach in Section IV
and discuss the adjustments for this more complex scenario. Finally, we present our findings in Section V and conclude in
Section VI.
Notations: BoldfacexandXdenote vectors resp. matrices.IN is anN×N identity matrix. We use the operators
Transpose{·}T, Hermitian{·}H, ExpectationE{·}, Hadamard product◦, absolute value|·|and Euclidean norm∥·∥.

II. DOWNLINKSATELLITECOMMUNICATIONMODEL
We examine a single-satellite multi-user downlink scenario as depicted in Fig. 1. TheLEOsatellite is equipped with a
Uniform Linear Array (ULA) consisting ofNantennas with an inter-antenna-distancednand transmit gainGSat. TheKusers
are assumed to be handheld devices with just one receive antenna and low receive gainGUsr. The Line-of-Sight (LoS) channel
hk∈C^1 ×Nbetween the satellite and userkis modeled by

```
hk(νk) =
```
#### 1

#### √

```
PLk
```
```
e−jκkvk(cos(νk)). (1)
```
The path lossPLkis the linear representation of the free space path lossFSPLkinfluenced by large scale fadingLFk∼N(0,σ^2 LF)
withPLdBk =FSPLdBk +LFdBk. The linear free space path lossFSPLkfor a given wavelengthλand a satellite-to-user distance
dkis

```
FSPLk=
```
```
16 π^2 d^2 k
λ^2 GUsrGSat
```
#### . (2)

The overall phase shift from the satellite to userkcorresponds toκk∈[0, 2 π], while the relative phase shifts from theN
satellite antennas to userkare determined by the steering vectorvk∈C^1 ×N. Then-th entry of the steering vectorvk(cos(νk))
calculates as follows

```
vkn(cos(νk)) =e−jπ
```
```
dλn(N+1− 2 n) cos(νk)
, (3)
```
whereνkis the Angle of Departure (AoD) from the satellite to userk. Under real circumstances, the estimate of theAoDsat the
satellite might be flawed. We model this behavior as a uniformly distributed additive errorεk∼U(−∆ε,+∆ε)on the space
anglescos(νk). In [8], we show that this error can be interpreted as an overall multiplicative errorvk(εk,m)∈C^1 ×Non our
channel vectorhk(νk):

```
h ̃k(νk,εk) =hk(νk)◦vk
```
#### 

```
εk
```
#### 

#### . (4)

In order to performSDMA, we calculate a precoding vectorwk∈CN×^1 for each userkbased on this estimate of the channel
vector ̃hk∈C^1 ×N. Different approaches to determine the precoding vectors are discussed in the subsequent sections. After
calculating the precoding vectorwk, the data symbolskof each userkis weighted with this precoding vectorwk. Taking
complex Additive White Gaussian Noise (AWGN)nk∼CN(0,σ^2 n)into account, the received signalykfor userkresults in

```
yk=hkwksk+hk
```
#### PK

```
l̸=kwlsl+nk. (5)
```
From (5), we get the Signal-to-Interference-plus-Noise Ratio (SINR)Γkfor userk

```
Γk=
```
```
|hkwk|^2
σn^2 +
```
#### PK

```
l̸=k|hkwl|
2
```
#### . (6)

The achievable rateR, equal to the sum rate, is given by

#### R=

#### PK

```
k=1log 2 (1 + Γk) (7)
```
and serves as the performance metric for the different precoding approaches in this paper. In summary, our goal is to maximize
the expected sum rate in presence of imperfect positional knowledge (4) by learning a robust precoding algorithm using the
SACtechnique.

III. CONVENTIONALPRECODERS
This section introduces 1) a conventional non-robustMMSEprecoding approach and 2) a robustSLNRprecoder based on
the second order statistics of the channel considering imperfect position knowledge. These analytical precoders serve as a
benchmark for the learned precoders.


## νk

## ν

## k− 1

## h hk

## k− 1

## d 0

## k

## k− 1

## Dk

Fig. 1. The single-satellite downlink scenario. The satellite is positioned at an altituded 0. Two usersk,k− 1 , are positioned atAoDsνk,νk− 1. They are
characterized by their channel vectorshk,hk− 1 and their inter-user distanceDUsr.

A. MMSE

TheMMSEprecoder is a well established precoder for scenarios with perfect channel state information [12] and will serve
as a baseline comparison in this work. For a channel estimationH ̃= [ ̃h 1 ... ̃hK]Tthe correspondingMMSEprecoding matrix
WMMSE= [wMMSE 1 ...wMMSEK ]is given as

#### WMMSE=

```
s
P
tr{W′HW′}
```
#### ·W′

#### W′=

```
h
H ̃HH ̃+σ^2 nK
P
```
#### IN

```
i− 1
H ̃H
```
#### , (8)

whereP denotes the overall transmit power of the satellite. We highlight that theMMSEapproach is not always optimal in
terms of the sum rateR(7).

B. Robust SLNR

The authors in [6] have recently introduced an analytical robust precoding approach for a channel and error model equivalent
to ours (4). Because the optimization of theSINRsis NP-hard [6], the authors of [6] maximize with regard to the instantaneous
SLNRsγkinstead, with

```
γk=
|hkwk|^2
σ^2 n+
```
#### PK

```
l̸=k|hlwk|
2
```
#### . (9)

Assuming equal power distribution among the users and considering the statistics of the estimation errors in the user positions,
the optimization problem corresponds to maximizing the meanSLNR ̄γk=E{γk}, i.e.,


```
Transforms Actorμ NormalizationTransforms & Calculation (7)Sum Rate
```
```
Memory
Buffer
```
```
Learning
Critic Module
Qˆ
```
```
H ̃t zt at Wt
```
```
Rt
Sampling
```
Fig. 2. The learnedSACprecoders’ process flow. Black arrows form an inference step, blue arrows show the components that are collected for a learning
step, and red arrows showNNparameter updates.

```
max
wk
```
#### E

#### (

```
|hkwk|^2
σ^2 n+
```
#### PK

```
l̸=k|hlwk|
2
```
#### )

```
s.t. wHkwk≤
```
#### P

#### K

#### . (10)

The precoding vectorwrSLNRk for each userk, which satisfies (10), has been derived in [6] as

```
wrSLNRk =
```
```
r
P
K
```
```
ψk,max, (11)
```
whereψk,maxis the eigenvector that corresponds to the largest eigenvalue of
X

```
l̸=k
```
```
σv^2 lRvl+σ^2 n
```
#### K

#### P

#### IN

#### − 1

```
σ^2 vkRvk, (12)
```
with the inverse path lossσv^2 k = 1/PLk andRvk∈CN×N being the autocorrelation matrix of the steering vectorsvk.
The autocorrelation matrix of the steering vectorsRvk=E{vkvHk}is used because it has similar characteristics to the
autocorrelation matrix of the channelE{hkhHk}, which is needed to solve (10). If we define an erroneous space angle
φˆk= cos(νk) +εkand use the definition of the steering vector from equation (3), we can rewrite the[n,n′]-th element
ofRvkas

```
[Rvk]n,n′=E
```
```
n
e−j
```
(^2) λπdn(n−n′)(φˆε−εk)o
(13)
=e−j
(^2) λπdn(n−n′)φˆε
φε(^2 λπdn(n−n′)), (14)
whereφεis the characteristic function that describes the probability distribution of the error. For a uniformly distributed error,
φεequals the sinc-function [6]
φε(t) =sinc(t∆ε). (15)
Because the above precoder is optimized with regard to the meanSLNRand not theSINR, it does not necessarily maximize the
sum rate (7). However, the authors in [6] show that, for perfect position knowledge and sufficiently large inter-user distances
DUsr, the robustSLNRprecoder is capacity achieving. We also note that this precoder always distributes the transmit powerP
evenly among theKusers, even though cases with varying path losses between the users are probable.

#### IV. REINFORCEMENTLEARNEDPRECODER

Our goal in this section will be to find a function that takes as input the estimated channel matrixH ̃ (4) and outputs a
precoding matrixWthat maximizes the expectationR ̄of the achievable sum rateR(7). We will use a parameterized deepNN
μθμ, subsequently called Actor Neural Network (AcNN), as a model function and then use theSAC[11] algorithm to tune this
network’s parametersθμ.SACassumes the true system dynamics are unknown, e.g., the distribution of errors on the estimated
CSITH ̃. Therefore, it trains a secondNNQˆθQˆthat has parametersθQˆin parallel to approximate the mapping of(H ̃,W)→R ̄.

This known functionQˆθQˆ, subsequently referred to as the Critic Neural Network (CrNN), is used to as a guide to tune the
precodingAcNNμθμ.SACrecommends the use of multiple, independently initializedCrNNfor stability [11]. In the following,

we will omit the parameter indicesθinAcNNμθμ≡μandCrNNQˆθQˆ≡Qˆfor readability.
Learning to optimize theNNparameters viaSACis comprised of two independent components: 1) data generation; 2) parameter
updates. Fig. 2 gives an overview of the process flow, we describe it in more detail in the following. First, to generate the data
to learn from, we perform an inference stept, starting by obtaining a complex-valued channel state estimateH ̃t∈CK×N. We
then flatten it into a vector and transform it to a real vector ̃z∈R^1 ×^2 KNas it is easier forNNto digest. For the real-complex
transformation we consider decomposition into a) real and imaginary part, as well as b) decomposition into magnitude and
phase, the choice of which we will discuss briefly later. The inputs ̃zare also standardized to approximately zero mean and
unit variance using static scaling factors, which is known to promote convergence speed inNN[10]. We discuss this choice


```
TABLE I
SELECTEDPARAMETERS
```
```
Noise Powerσn^26 e- 13 W Transmit PowerP 100 W
Satellite Altituded 0 600 km Antenna Nr.N { 10 , 16 }
User Nr.K 3 Overall Sat. Gain 20 dBi
Wavelengthλ 15 cm Gain per UserGUsr 0 dBi
Inter-Ant.-Distancedn 3 λ/ 2 SGDOptimizer Adam
Training batch sizeB 1024 Init. LRCrNN,AcNN 1 e- 4 , 1 e- 5
Learning Buffer Size 100 000 Inference / Learning 10 : 1
L2 ScalesαQˆ,αμ 0. 1 log Entropy Scaleαe Var.
```
in Section V-A. We forward the standardized vectorzt∈R^1 ×^2 KN through theAcNN, which is a standard feed-forwardNN
with four times as many outputs as we require entries for a precoding matrixW. The outputs are grouped in pairs of two,
where one of each pair represents the mean and the other the scale of a Normal distribution to sample from. This formulation
gives us a measure of uncertainty about each output that we will make use of during the training step. After sampling from
the distributions given by the output pairs, we transform this outputa∈R^1 ×^2 KNinto a complex vector of half length and
reshape to a precoding matrix. Finally, we rescale the matrix to the available signal power and gain a normalized precoding
matrixWt∈CN×K. In testing, we find that magnitude-phase decomposition works best for the network input and real-
imaginary composition works best for the output, though this is still under investigation. To conclude a data generation step,
we evaluate the sum rateRtachieved by this precoding matrixWtand store the tuple of(z,a,R)tin a memory buffer for
the learning step.
Learning steps are performed using the Stochastic Gradient Descent (SGD) principle of noisily approximating a gradient step
based on a batch subset of the entire data set. During a learning step, first, a batchB={(·)b|b∼ U( 0 ,B ̄)}of|B|=B
tuples are drawn from the memory buffer of sizeB ̄. Next, theCrNNsQˆare updated. A loss is calculated as follows,

```
LQˆ=
```
#### 1

#### B

#### X

```
b∈B
```
```
(Qˆ(zb,ab)−Rb)^2 +αQˆ∥θQˆ∥^2 , (16)
```
where the first term describes the mean square error in sum rate estimation and the second term is a weight regularization
that is discussed later.αQˆis a scaling term. ASGD-like update is performed to minimize this batch loss. Next, theAcNNis
updated. Its loss contains three sum terms:

```
Lμ=
```
#### 1

#### B

#### X

```
b∈B
```
```
−Qˆ(zb,μ(zb)) (17)
```
#### +

#### 1

#### B

#### X

```
b∈B
```
```
exp(αe) log(π(μ(zb))) (18)
```
```
+αμ∥θμ∥, (19)
```
whereαe,αμare scaling terms. Recalling that the precoder is sampled from a distribution parameterized by the outputs of the
AcNN,π(·)is the probability of the sampled precoding, given this distribution. The first term calls the optimization to maximize
the estimated sum rate. The second term encourages the optimization to increase the output variance where it does not foresee
gains in sum rate from keeping the variance tight. It thereby encourages theAcNNto explore more thoroughly where no good
solution has been found yet. The third term is, again, a weight regularization. If multipleCrNNare used, we take the minimum
sum rate estimate per tuple for a conservative estimate.
In the following section, we discuss specific implementation choices such as the weight regularization, and then proceed to
evaluate the learned precoder and the two benchmarks.

V. EVALUATION
Here, we discuss specific implementation details that we found crucial in learning a precoder. We then present performance
comparisons and discuss the relative advantages and disadvantages of the precoders under study. The full code implementation,
the trained models, their training configurations and supplemental figures are available online [13]. Table I lists a selection of
important system and learning parameters.

A. Implementation Details

As discussed in the previous section, we find rescaling the network inputs to be highly beneficial for fast and stable
learning, in accordance with theoretical literature [10]. It could be argued that taking a large number of samples to find the
population means and scales before starting to learn is sample inefficient and undesirable, however, doing a hypothesis test at a
significance level of5 %, we find the statistics of the population to be approximable within±10 %by taking just 100 samples.


For similar reasons, rescaling values not just at the input but also between network layers is desirable and proves beneficial,
with intermediate Batch Normalization layers [14] being the most common approach. OurNNimplementations correspondingly
use four fully connected layers ( 512 nodes each) stacked alternatingly with Batch Normalization layers.
Another critical change to our prior work is the addition of weight regularization terms in (16), (17). Weight regularization
is another standard method, the reasons for its positive influence still being under intense scrutiny [15]. We find weight
regularization to be specifically favorable for training in the presence of error, which intuitively can be interpreted as stopping
a learning step from committing overmuch to a single batch of channel realizations. In a similar vein, we significantly increase
the number of experiences held in the memory buffer compared to our earlier work ( 10 k→ 100 k) to provide a more rich set of
training experiences to sample from. To compensate, we also adjust the ratio of inference steps per learning step (1 : 1→10 : 1)
so that the age of the oldest experience in the buffer stays the same, as per [16].
Overall, we find that the most significant parameter choice is theSGDlearning rate, which we keep variable with a Cosine
Decay schedule in the area of 1 e- 4 , 1 e- 5 forCrNN,AcNNrespectively. For the detailed configuration we again refer to the
repository [13]. We also highlight that, considering practicality, hyper parameter search and training time are limited, thus, our
trained models certainly do not present the optimum solution. Training is performed until no significant performance increase
is observed, up to around 1 e 6 to 14 e 6 simulation stepstdepending on the scenario.

B. Evaluation Design

Each simulation will assume a certain mean user distanceD ̄Usr. For each simulation stept, each user will be assigned a
new position uniform randomly within±D ̄Usr/ 2 around their mean position, and channel conditions will be updated according
to Section II. We investigate the following three scenarios: a)N= 10satellite antennas, mean user distanceD ̄Usr= 100 km;
b)N= 16,D ̄Usr= 100 km; c)N= 16,D ̄Usr= 10 km. Before evaluation, we train two learned precoders on each scenario:
1) trained with perfectCSIT, marked with blue square e.g.,pSAC1; 2) trained at error bound∆ε= 0. 05 , marked with green
cross, e.g.,rSAC1. Evaluations are repeated with 1000 Monte Carlo iterations to account for the stochastic elements of the
simulation design, we evaluate the mean performance and its standard deviation.

C. Results

We first explore scenario a), where a mean user distanceD ̄Usr= 100 kmensures, on average, good spatial partitioning of
the users andN= 10satellite antennas will produce wide beams relative to user distances. We train two precoders,pSAC
with perfectCSITandrSAC1with erroneousCSIT, and then compare their mean performance when evaluated on increasingly
unreliableCSIT. Fig. 3 presents the mean performance at increasingly large settings of error bound∆εfor the two trained
schedulers as well as theMMSEand robustrSLNRprecoders from Section III. As expected, all precoders’ performances are
impacted by increasingly unreliableCSIT. PrecodersMMSE, robustrSLNRandpSAC1achieve comparable performance for perfect
CSIT(∆ε= 0. 0 ), whilerSAC1, having only encountered unreliableCSITduring training, has adopted a strategy that does not
scale as well with reliable information. On the other hand,rSAC1shows the least performance degradation as theCSITbecomes
increasingly unreliable. It takes the performance lead at its training point of∆ε= 0. 05 , with the performance gap increasing
thereafter. TheMMSEprecoder, expectedly, shows the worst performance with unreliableCSIT, while the robustrSLNRandpSAC
precoders are closely matched. This result might, however, be slightly misleading, as robustrSLNRandpSAC1have adopted
significantly different strategies, which we will see in the following.
We repeat this experiment for scenario b), where the increased number ofN= 16satellite antennas allows for more narrow
beams. This enables better user separation even at close user distances, at the cost of more severe performance drops when
a beam is missteered. The results are displayed in Fig. 4 and follow the same trajectory as scenario a), though we see that
the narrower beams lead to a more pronounced performance drop as the unreliability∆εincreases. In order to understand
how each precoder achieves their robustness, we take a look at their beam patterns for a specific simulation realization. Fig. 5
compares the beam patterns of therSLNRand our learned precoderpSAC2with theMMSEprecoder. We select this plot as a
representative for the precoders’ behavior, though we highlight that it depicts just one realization of user positions, error values,
large scale fading. Further beam patterns are provided in [13]. Black dots and dotted lines represent the true user positions,
whereas the erroneously estimated positions can be discerned by the placement of theMMSEprecoder’s beams. We observe
in the top figure that, compared to theMMSEprecoder, the robustrSLNRprecoder achieves its robustness by trading beam
height in favor of beam width, covering a larger area. On the contrary, thepSAC3precoder in the bottom figure, not having
encountered unreliable information during training, has no incentive to opt for wider beams. Nevertheless, we observe that it
achieves robustness by two other factors: 1) power allocation among the different user’s beams; 2) better user tracking. We
report that it achieves the second factor by exploiting the power fading information of theCSIT. In the depicted realization, the
center user’s large scale fading is near one, hence, the power fading of theCSITclosely corresponds to a certain satellite-to-user
distance that the learned precoder uses to fine-tune its beam positioning. In Fig. 6, we repeat this comparison for therSAC
precoder that was trained for robustness to severe errors. We observe that this scheduler makes use of more irregular beam
shapes to cover wide areas.


```
0. 00 0. 02 0. 04 0. 06 0. 08 0. 10
Error Bound∆ε
```
```
0
```
```
2
```
```
4
```
```
Sum Rate
```
```
R
```
```
[bit/s/Hz]
```
```
MMSE
rSLNR
```
```
pSAC
rSAC
```
Fig. 3. Scenario a),N= 10satellite antennas,D ̄Usr= 100 kmmean user distance. Testing precoders mean performance with increasing error bounds∆ε.
pSAC1is trained with perfectCSIT,rSAC1is trained at∆ε= 0. 05. Markers on the horizontal axis show the error bound thatpSAC1resp.rSAC1were trained
at.

```
0. 00 0. 02 0. 04 0. 06 0. 08 0. 10
Error Bound∆ε
```
```
0
```
```
2
```
```
4
```
```
Sum Rate
```
```
R
```
```
[bit/s/Hz]
```
```
MMSE
rSLNR
```
```
pSAC
rSAC
```
Fig. 4. Scenario b),N= 16satellite antennas,D ̄Usr= 100 kmmean user distance. Testing precoders mean performance with increasing error bounds∆ε.
pSAC2is trained with perfectCSIT,rSAC2is trained at∆ε= 0. 05. Markers on the horizontal axis show the error bound thatpSAC2resp.rSAC2were trained
at.

```
0
```
```
500
```
```
Power Gain
```
```
Users
MMSE
rSLNR
```
```
1. 3 1. 4 1. 5 1. 6 1. 7 1. 8
Angle of Departureν
```
```
0
```
```
500
```
```
Power Gain
```
```
Users
MMSE
pSAC
```
Fig. 5. Beam patterns for one specific simulation realization of scenario b). Curves represent the precoding vectors of the different users at differentAoDsν
in linear scale. Sum ratesRachieved:MMSE: 0 .83 bit/s/Hz,rSLNR: 1 .33 bit/s/Hz,pSAC2: 2 .15 bit/s/Hz.

```
0
```
```
500
```
```
Power Gain
```
```
Users
MMSE
rSLNR
```
```
1. 3 1. 4 1. 5 1. 6 1. 7 1. 8
Angle of Departureν
```
```
0
```
```
500
```
```
Power Gain
```
```
Users
MMSE
rSAC
```
Fig. 6. Beam patterns for one specific simulation realization of scenario b). Curves represent the precoding vectors of the different users at differentAoDsν
in linear scale. Sum ratesRachieved:MMSE: 0 .86 bit/s/Hz,rSLNR: 1 .54 bit/s/Hz,rSAC2: 2 .07 bit/s/Hz.

Finally, in scenario c), we study a case in which theMMSEandrSLNRprecoders are not sum rate optimal due to very
close average user positioning (D ̄Usr= 10 km) relative to the beam width. Fig. 7 displays again a performance sweep over


```
0. 00 0. 02 0. 04 0. 06 0. 08 0. 10
Error Bound∆ε
```
```
0
```
```
1
```
```
2
```
```
3
```
```
4
```
```
Sum Rate
```
```
R
```
```
[bit/s/Hz] MMSE
rSLNR
```
```
pSAC
rSAC
```
Fig. 7. Scenario c),N= 16satellite antennas,D ̄Usr= 10 kmmean user distance. Bad conditions forMMSEandrSLNRprecoders. Testing precoders mean
performance with increasing error bounds∆ε.pSAC3is trained with perfectCSIT,rSAC3is trained at∆ε= 0. 05. Markers on the horizontal axis show the
error bound thatpSAC3resp.rSAC3were trained at.

increasingly unreliableCSITfor this scenario. We see the learned precoderspSAC3,rSAC3attain greater sum rate performances,
achieved by simply not allocating any power to the center user such that it does not interfere with the other two users. We also
see that the robustrSLNRprecoder, with its preassumptions violated by the high spatial coupling of user channels, is not able
to provide robustness in this scenario even compared to theMMSEprecoder. The learned precoders achieve very high degrees
of robustness, though we must qualify this result stemming from the erroneous channel realizations being much larger than the
variation through user positioning in this scenario. The learned precoders exploit this by discarding implausibleCSIT, which
we do not expect to be possible to the same degree under real circumstances.
In summary, learned precoders offer high flexibility, adjusting to various combinations of user constellations, satellite
configurations and error influences, while achieving strong performance. Analytical precoders, while perhaps more predictable,
can be complex and costly to solve mathematically and may suffer greatly when the real conditions do not match those that
the precoder was modeled on.

#### VI. CONCLUSIONS

In this paper we studied the influence of imperfect user position knowledge onSDMAprecoding inLEOsatellite downlink
scenarios. Our goal was to design a robust precoding approach that manages inter-user interference for arbitrary error sources
and channels. Using data-driven deepRLvia theSACalgorithm, we built learned precoders that obtained high performance in
terms of both achievable rate and robustness to positioning errors. We qualified these results in comparison to two analytical
benchmark precoders, the conventionalMMSEprecoder as well as a robust precoder that leverages stochastic channel information.
Their flexibility at high performance could make learned satellite precoding algorithms an attractive candidate for use in 6G
and beyond.

#### REFERENCES

[1] 3GPP TR 38.863, “Technical specification group radio access network; solutions for nr to support non-terrestrial networks (NTN): Non-terrestrial networks
(NTN) related RF and co-existence aspects (release 17),” Sep. 2022.
[2] I. Leyva-Mayorga, B. Soret, M. R ̈oper, D. Wubben, B. Matthiesen, A. Dekorsy, and P. Popovski, “LEO Small-Satellite Constellations for 5G and ̈
Beyond-5G Communications,”IEEE Access, vol. 8, pp. 184 955–184 964, 2020.
[3] Z. Qu, G. Zhang, H. Cao, and J. Xie, “LEO Satellite Constellation for Internet of Things,”IEEE Access, vol. 5, pp. 18 391–18 401, 2017.
[4] M.A. V ́ ́azquez, M. B. Shankar, C. I. Kourogiorgas, P.-D. Arapoglou, V. Icolari, S. Chatzinotas, A. D. Panagopoulos, and A. I. P ́erez-Neira, “Precoding,
Scheduling and Link Adaptation in Mobile Interactive Multibeam Satellite Systems,”IEEE J. Sel. Areas Commun, vol. 36, no. 5, pp. 971–980, 2018.
[5] Y. Liu, Y. Wang, J. Wang, L. You, W. Wang, and X. Gao, “Robust Downlink Precoding for LEO Satellite Systems With Per-Antenna Power Constraints,”
IEEE Trans. Veh. Technol., vol. 71, no. 10, pp. 10 694–10 711, 2022.
[6] M. Roper, B. Matthiesen, D. W ̈ ubben, P. Popovski, and A. Dekorsy, “Robust Precoding via Characteristic Functions for VSAT to Multi-Satellite Uplink ̈
Transmission,”arXiv:2301.12973, 2023.
[7] A. Schr ̈oder, M. Roper, D. W ̈ ̈ubben, B. Matthiesen, P. Popovski, and A. Dekorsy, “A Comparison between RSMA, SDMA, and OMA in Multibeam
LEO Satellite Systems,” in26th WSA & 13th SCC, 2023.
[8] S. Gracla, A. Schroder, M. R ̈ oper, C. Bockelmann, D. W ̈ ̈ubben, and A. Dekorsy, “Learning Model-Free Robust Precoding for Cooperative Multibeam
Satellite Communications,” in2023 IEEE ICASSPW, 2023.
[9] H. Dahrouj, R. Alghamdi, H. Alwazani, S. Bahanshal, A. A. Ahmad, A. Faisal, R. Shalabi, R. Alhadrami, A. Subasi, M. T. Al-Noryet al., “An Overview
of Machine Learning-Based Techniques for Solving Optimization Problems in Communications and Signal Processing,”IEEE Access, vol. 9, 2021.
[10] I. Goodfellow, Y. Bengio, and A. Courville,Deep Learning. MIT Press, 2016, [http://www.deeplearningbook.org.](http://www.deeplearningbook.org.)
[11] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, and S. Levine, “Soft Actor-Critic Algorithms
and Applications,”arXiv:1812.05905, Jan. 2019.
[12] S. Chatzinotas, G. Zheng, and B. Ottersten, “Energy-efficient MMSE beamforming and power allocation in multibeam satellite systems,” inIEEE
ASILOMAR, 2011.
[13] A. Schroder and S. Gracla, “Learning Beamforming 2,” https://github.com/Steffengra/2310 ̈ beamforminglearner2, 2023.
[14] R. Balestriero and R. G. Baraniuk, “Batch Normalization Explained,” Sep. 2022, arXiv:2209.14778.
[15] M. Andriushchenko, F. D’Angelo, A. Varre, and N. Flammarion, “Why Do We Need Weight Decay in Modern Deep Learning?” Oct. 2023,
arXiv:2310.04415.
[16] W. Fedus, P. Ramachandran, R. Agarwal, Y. Bengio, H. Larochelle, M. Rowland, and W. Dabney, “Revisiting Fundamentals of Experience Replay,” in
37th PMLR, 2020.


