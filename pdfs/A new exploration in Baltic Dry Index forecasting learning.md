Vol.:(0123456789)
Maritime Economics & Logistics (2024) 26:21–43
https://doi.org/10.1057/s41278-023-00278-6
ORIGINAL ARTICLE
A new exploration in Baltic Dry Index forecasting learning: 
application of a deep ensemble model
Miao Su1 · Keun Sik Park2 · Sung Hoon Bae2,3
Accepted: 16 November 2023 / Published online: 22 December 2023 
© The Author(s), under exclusive licence to Springer Nature Limited 2023
Abstract
World trade is growing constantly, facilitated by the fast expansion of logistics. 
However, risks and uncertainty in shipping have also increased, in dire need to be 
addressed by the research community, through more accurate and efficient meth-
ods of forecasting. In recent years, combining attention models and deep learning 
has produced remarkable results in various domains. With daily data spanning the 
period from January 6, 1995, to September 16, 2022 (totaling 6896 observations), 
we predict the Baltic Dry Index (BDI) using a deep integrated model (CNN-BiL -
STM-AM) comprising a convolutional neural network (CNN), bi-directional long 
short-term memory (BiLSTM), and the attention mechanism (AM). Our findings 
indicate that the integrated model CNN-BiLSTM-AM encompasses the nonlinear 
and nonstationary characteristics of the shipping industry, and it has a greater pre-
diction accuracy than any single model, with an R 2 value of 96.9%. This research 
shows that focusing on the data’s value has a particular appeal in the intelligence 
era. The study enhances the integrated research of machine learning in the shipping 
business and offers a foundation for economic decisions.
Keywords Logistics · Shipping business · Forecasting · Baltic Dry Index · Machine 
learning · Shipping economics
 * Sung Hoon Bae 
 sunghoonbae@cau.ac.kr
1 Graduate School of Technology Management, Kyunghee University, Seoul, Korea
2 Department of Trade and Logistics, Chung Ang University, 84 Heukseok-ro, Dongjak gu, Seoul, 
Korea
3 Samsung SDS, Seoul, Korea

22 M. Su et al.
1 Introduction
The shipping sector has made a vital contribution to the economic growth of many 
countries and it plays an essential part in the world trade system (Kanamoto et al. 
2021). Shipping transports 80% to 90% of international trade volume and is thus 
one of the most important transport modes in the world (Zhang et  al. 2019). The 
industry’s volatility indicators are composite economic indicators (Duru et al. 2012; 
Chang et  al. 2014). Among the many, the Baltic Dry Index (BDI), published by 
the Baltic Exchange, is a key price (freight rate) indicator (Cullinane et  al. 1999; 
Lin et al. 2019). BDI is often considered as the “barometer” of the shipping indus-
try, as well as a leading indicator of general economic activity (Liu et  al. 2022). 
Specifically, the BDI analyzes transportation prices for dry bulk cargo such as iron 
ore, coal, and grains from producing countries such as Brazil, Australia, and South 
Africa to markets in the United States, Europe, Japan, and China (Duru et al. 2012).
As a benchmark for measuring transportation costs, researchers have often 
considered the BDI as an ideal proxy variable to global economic activity. 
Changes in BDI have been seen as a leading measure of future economic growth 
or contraction (Lin and Sim 2013; Gu et  al. 2019; Haralambides 2019). In 
finance, investment bankers and hedge fund managers commonly view the BDI 
as a leading economic indicator for financial markets such as equities and com-
modities (Alizadeh and Muradoglu 2014). The BDI is also a crucial indicator for 
bulk carrier owners, charterers, and shippers when making crucial decisions or 
implementing strategic business plans (Chou and Lin 2019; Yang and Mehmed 
2019), because the index can assist stakeholders with maritime business planning 
and market risk avoidance. Therefore, an accurate prediction of the BDI provides 
a decision basis for decision-makers in the shipping sector, industry, finance, and 
government agencies and it is beneficial to the healthy growth of the global eco-
nomic system (Chen et al. 2012; Apergis and Payne 2013; Wang et al. 2015).
In the past 40  years, the BDI has been an important research area (Duru and 
Yoshida 2009). There are three primary aspects in BDI research. First, BDI is used 
by researchers as an indirect indicator of market expectations regarding the global 
economy and trade. Authors use it as a predictor of various economic variables such 
as exchange rate fluctuations, inflation, and demand for transport of raw materials, 
energy, and agricultural products (Bildirici et  al. 2015; Papailias et  al. 2017; Han 
et al. 2020). The quantitative relationship between the BDI and commodity markets 
constitutes the second area of study (Lin et al. 2019; Said and Giouvris 2019; Ban-
dyopadhyay and Rajib 2023). The third aspect concerns BDI forecasting.
The BDI is a typical nonstationary nonlinear time series influenced by numer -
ous complex variables (Liu et al. 2022). Although the literature provides various 
studies and analyses of BDI trends, the time-varying and unpredictable character 
of the shipping industry makes it difficult, if not impossible, to predict it with 
a satisfactory degree of accuracy. Freight rate forecasting is unquestionably an 
essential albeit difficult endeavor (Liu et al. 2022).
In summary, the logistics industry urgently requires effective data mining 
techniques. For example, earlier BDI prediction research used predominantly 

23
A new exploration in Baltic Dry Index forecasting learning:…
causality-based econometric models which, however, required the regularity 
and stationarity of large datasets (Duru 2010; Duru et al. 2012; Gao et al. 2023). 
Nonetheless, the data features of the BDI cause significant mistakes in the pre-
cision of econometric models (Zhang et al. 2018). However, research has dem-
onstrated that AI-based methods yield many superior outcomes compared to 
conventional econometric and statistical models. Consequently, researchers have 
utilized numerous nonlinear and artificial intelligence (AI) techniques, such as 
artificial neural networks (ANN), support vector machine (SVM), and nonlinear 
regression (Yang et al. 2008; Chou and Lin 2019). However, the limited nature 
of a single AI prediction approach makes it susceptible to over- or undertrain-
ing, resulting in errors. Research has thus questioned a single machine learning 
method (Zhang et al. 2019). In addition, most previous studies viewed BDI pre-
diction as a forecast of a single-day variable (Kamal et  al. 2020). But markets 
need projections for at least one week in advance, given the volatility of freight 
rates (Bae et al. 2021). In conclusion, there are still significant discrepancies in 
the BDI research, particularly in BDI prediction.
This study, therefore, collected daily prices from January 6, 1995, to September 
16, 2022, totaling 6896 observations. We employ an integrated model (CNN-BiL -
STM-AM) comprising a convolutional neural network (CNN); bi-directional long 
short-term memory (BiLSTM); and attention mechanism (AM), to increase the 
accuracy of BDI freight rate forecasting.
Among the many contributions of this study is our novel approach based on deep 
learning ensemble models that effectively capture the nonlinear and nonstationary 
properties of freight rates. This study also expands the scope of forecasting meth-
odologies. Second, BDI forecasting studies often include daily forecasts. This study 
increases the forecasting horizon and enhances the forecasting cycle, maintaining 
forecast accuracy.
The remainder of this study is as follows. Section 2 reviews the methods used to 
forecast BDI time series, machine learning, and forecasting combined methods. Sec-
tion 3 presents our data and the ensemble methods used here. Section 4 discusses the 
main empirical findings and compares them with alternative models. Finally, Sect. 5 
discusses the research results, concludes, and suggests further research avenues.
2  Literature review
2.1  Econometric models
Freight rate modeling and forecasting research has taken place for more than half a 
century (Duru 2010). The most prevalent methods have been econometric modeling 
or machine learning, which include time series, linear regression, fuzzy sets, and 
artificial neural networks (Zhang et al. 2018; Chou and Lin 2019). There is wide-
spread usage of linear regression models for predictive analysis in the context of 
price volatility (Bae et al. 2021). Models of autoregressive integrated moving aver -
age (ARIMA), vector autoregressive (VAR), generalized autoregressive conditional 
heteroscedasticity (GARCH), and vector error correction (VEC) are examples of 

24 M. Su et al.
econometric techniques (Bildirici et al. 2015; Zhang et al. 2019; Xiao et al. 2021). 
Initially, a study developed a VAR model for predicting the BDI using a coin inte-
gration method and a unit root test for the time series (Veenstra and Franses 1997). 
Another study introduced the ARIMA model to dry bulk freight rate forecasting for 
the first time, to forecast the impact of freight rates on the shipping industry (Cul-
linane et al. 1999). Academics have readily understood the significance of applying 
several econometric models to freight rate forecasting.
Using autoregressive integral moving average and VAR models, Kavussanos 
and Alizadeh (2001) examined the seasonal characteristics of the dry bulk shipping 
industry. However, another study using ARIMA and VAR to anticipate freight rates 
for multiple dry bulk routes indicated that VAR outperforms ARIMA in out-of-
sample forecasting (Chen et al. 2012). In addition, Tsioumas et al. (2017) validated 
the initial research findings. They propose a multivariate VAR model (VARX) using 
exogenous variables and discover that the VAR model outperforms the ARIMA 
technique. In addition, these researchers discussed the performance of VAR, 
ARIMA, and vector balance correction (VECM) approaches for predicting spot and 
forward freight rates. They discovered that VAR and VECM predicted spot freight 
rates somewhat better than ARIMA (Batchelor et al. 2007). Moreover, Kavussanos 
and Alizadeh (2001) discovered that the VECM model predicts freight rates more 
accurately than the ARIMA model.
The majority of these classic statistical analyses include error measures such as 
the mean absolute percentage error (MAPE), the mean square error (MSE), and the 
root mean square error (RMSE) to improve the prediction accuracy. Nonetheless, 
such conventional methods produce many inconsistent results because of the short-
comings of statistical extrapolation. Specifically, linear regression forecasting mod-
els, based on mathematical equations, cannot learn dynamically; predict patterns of 
change; or identify moving trends (Lin et  al. 2007). In addition, traditional linear 
regression models and time series methods are characterized by data intensity, noise, 
non-stationarity, high uncertainty, and implied relationships in the actual forecasting 
process. This can reduce the accuracy and stability of model predictions (Chou and 
Lin 2019). Therefore, research has gradually shifted focus to AI techniques to inves-
tigate the inherent complexity of the BDI (Zhang et al. 2019).
2.2  Machine learning models
Freight market forecasting outcomes have had a poor reputation before machine 
learning advances, due to the difficulty of capturing the hidden nonlinear proper -
ties of the dry bulk freight index with conventional econometric methods (Zhang 
et al. 2018, 2019; Katris and Kavussanos 2021). As a result, academics have increas-
ingly focused on various AI techniques. The SVM, ANN nonlinear regression and 
fuzzy system are excellent machine learning-based prediction approaches that have 
emerged in recent years (Kamal et al. 2020; Xiao et al. 2021). First, support vector 
machines have great generalization and nonlinear function approximation capabili-
ties. Thus, Yang et al. (2008) used SVM to analyze and forecast freight rates, such 
as those of the China Coastal Bulk Freight Index (CCBFI), China Containerized 

25
A new exploration in Baltic Dry Index forecasting learning:…
Freight Index (CCFI), and Baltic Freight Index (BFI). In addition, Guan et al. (2016) 
developed a forecasting model based on SVM and produced detailed projections 
using the dry bulk market’s BSI data (Baltic Supramax Index).
In addition, BDI-related prediction research frequently uses artificial neural net-
work models. One study first proposed a decomposition technique for BDI data and 
then used neural networks for prediction (Zeng et al. 2016). Another study devel-
oped three artificial neural network forecasting techniques using historical BDI data 
and global crude oil prices. The findings demonstrated that while the three neural 
network approaches perform similarly, ANN’s prediction effect is superior (Şahin 
et  al. 2018). Other studies have used wavelet neural networks, deep neural net-
works, and fuzzy neural networks to analyze and forecast the BDI in greater detail 
(Leonov and Nikolov 2012; Chou and Lin 2019; Kamal et al. 2020). These studies 
have increased the popularity of neural networks in researching dry bulk freight rate 
forecasting.
Zhang et al. (2018) compared the artificial neural network model with the econo-
metric model. Their findings demonstrate that utilizing neural networks to predict 
the BDI is more accurate than conventional techniques. It is easier for neural net-
work-based predictive models to handle complex nonlinear interactions and give 
accurate estimates in this area. Determining the model structure of AI algorithms, 
however, is challenging. This is susceptible to overtraining or undertraining, both 
of which can result in deficiencies, such as the sensitivity to initial values causing it 
to fall into local minima (Zhang et al. 2019). Furthermore, neural networks are par -
ticularly vulnerable to undertraining and overtraining. This vulnerability may result 
in shortcomings, such as the issue of becoming stuck in local minima due to starting 
value sensitivity (Zhang et al. 2018).
Consequently, several authors have discussed novel approaches to forecasting. 
For instance, a number of them believe that the fuzzy-Delphi adjustment method 
enhances the precision of the statistical forecast of the dry bulk shipping index, and 
it has produced positive results (Duru et al. 2012). In addition, deep learning is a 
potential technique for reliable BDI prediction (Kamal et al. 2020). Among the deep 
learning methods, researchers consider the long short-term memory model a practi-
cal approach for dealing with dry bulk cargo time series problems. However, this 
method has not received much attention in BDI forecasting research (Xiao et  al. 
2021). To its credit, the AI-based strategy has provided many superior outcomes 
compared to conventional econometric and statistical models. Nevertheless, the 
results of the current research on BDI forecasting are diverse. However, we still do 
not comprehend the complementarity, benefits, and drawbacks of various prediction 
algorithms and more research is required into deep learning or integration models.
2.3  Integration models
The BDI time series is typically volatile and nonstationary. As a result, the ‘skills’ of 
forecasting tools, including econometric models and AI-based approaches, have lim-
itations. In addition, it has been shown that we cannot make statistically valid predic-
tions using only these techniques (Kamal et al. 2020) and newer research is looking 

26 M. Su et al.
into using integrated approaches to predict the BDI and improve prediction accuracy 
(Zhang et  al. 2019). Indeed, various disciplines have widely employed ensemble 
learning to enhance model performance by increasing the prediction capacity of a 
single model (Xiao et al. 2021). Research has suggested combining noise-reduction 
methods with AI-based algorithms as a standard method for constructing ensemble 
models (Zhang et al. 2019). For this reason, authors employ a wavelet transform to 
remove noise from BDI data (Han et al. 2014). Researchers also use empirical mode 
decomposition (EMD) and ANN to create a separate integrated model for BDI pre-
diction, achieving good accuracy (Zeng et al. 2016).
In another approach, Uyar et  al. (2016) used genetic algorithms with a well-
trained recurrent neural network to enhance the precision of long-term BDI forecast-
ing. In addition, Zhang et al. (2019) predicted the BDI using a method that com-
bined a dynamic fluctuation network (DFN) with AI. The authors found that the 
DFN-AI model had higher accuracy and trend-matching rate than its AI-based coun-
terpart. Further, fusing several independent models into a single ensemble model 
can enhance the prediction accuracy of the BDI. In addition, Kamal et  al. (2020) 
employed a deep integrated recurrent network consisting of a recurrent neural net-
work (RNN), long short-term memory (LSTM), and gated rectifier unit neural net-
work (GRU) to enhance predictive performance. The results of the ensemble mod-
els mentioned above indicate that hybrid models are superior to single AI models 
in time series forecasting. In reality, the hybrid forecasting approach combines the 
characteristics of each individual model and allows the respective benefits to com-
pensate for the shortcomings of other methods (Zhang et al. 2019). Consequently, 
we predict that integrated forecasting models will become a focal point in future 
maritime freight market analyses. However, the computational requirements of 
hybrid methods are intricate and call for further innovative research.
3  Method
3.1  Data description
From January 6, 1995, to September 16, 2022, daily pricing data of BDI were obtained 
from Clarksons. Figure 1 displays the trend. The series does not appear to be station-
ary. The data exhibit significant fluctuations, high frequency, and distinctive character-
istics, and the graph reveals major turning points at 2003, 2005, 2008, and 2021. Data 
points show substantial fluctuations at various time intervals. From 1999 to 2003, the 
BDI varied somewhat. Then, beginning in 2003, market volatility intensified. In 2008, 
the BDI reached a record high of 11,793. However, following the financial crisis of 
2008–2009, a flood of newbuildings entered the market resulting in an excess supply 
of tonnage. In conjunction with a decline in demand, this caused the Baltic Dry Index 
(BDI) to plummet to one of its lowest levels (Oomen 2012), falling dramatically to 700 
index points. Since then, the BDI has fluctuated continuously within a limited range. 
In February 2016, the index plummeted again to a 30-year low of 290. The index saw 
a brief recovery during the summer of the same year but subsequently proceeded to 
decline due to an oversupply of tonnage and a slower growth in commodities demand. 

27
A new exploration in Baltic Dry Index forecasting learning:…
As of 2019, the outbreak of the COVID-19 pandemic had a significant impact on ship-
ping, resulting in substantial variations in BDI (Notteboom and Haralambides 2020; 
Cullinane and Haralambides 2021).
Table 1 presents the descriptive statistics of the data, including mean, standard devi-
ation, skewness, kurtosis, and coefficient of variation. The skewness and kurtosis of the 
BDI series and its rate-of-change series are greater than zero. Furthermore, the stand-
ard deviation is high at 86.65% of the mean, indicating a high level of volatility in the 
freight market. In addition, the distribution of the series appears to be right-skewed, 
with evidence of fine kurtosis.
We assessed the BDI series using the Jarque–Bera test (Cromwell et al. 1994) and 
tested autocorrelation using Ljung and Box’s (1978) test. Finally, we determined sta-
tionarity using the Augmented Dickey-Fuller (ADF) test (Fuller 2009). Based on these 
results, the BDI series exhibits non-normality and autocorrelation.
3.2  Model
We predict the BDI using the CNN-BiLSTM-AM ensemble model, which includes a 
convolutional neural network, a bi-directional long short-term memory, and an atten-
tion mechanism. Using CNN, one extracts characteristics from the data inputs. The 
BiLSTM then uses these features, extracted by the CNN from the data input, for BDI 
prediction. In addition, the AM captures the effect of significant past temporal feature 
states, to increase the accuracy of predictions.
Fig. 1  Baltic Dry Index (BDI raw data)
Table 1  Descriptive statistics Mean value Standard deviation Skewness Kurtosis Variation
2116.47 1833.95 2.45 6.86 0.87

28 M. Su et al.
3.2.1  Convolutional neural networks (CNN)
Convolutional neural networks are one of the most prominent deep learning algo-
rithms. CNN is a feed-forward neural network with convolution calculation and a 
deep structure, highly suited to processing one-dimensional time domain sequences 
and picture data. Compared to other neural networks, CNN can efficiently and rap-
idly learn raw data and representations. Therefore, authors use it in various applica-
tions, including computer vision and natural language processing (Gu et al. 2018). 
The data convoluted in this study is 1D time series data, and 1D-CNN can extract 
the features of the data. The CNN model structure mainly includes input, convo-
lution, pooling, fully connected, and output layers. We use a single convolutional 
layer to reduce data dimensionality and extract serial information. Figure  2 shows 
the model’s structure.
Assuming a time series (xt1
,xt2
,… xtk
) of length k, after the convolution operation 
of n convolution kernels with stride = 1, we can obtain the output vector sequence 
(o1 ,o2 ,… ok−2 ) . The calculation process is:
3.2.2  Bi‑long short‑term memory network (BiLSTM)
The long short-term memory network is a deep learning model widely used in time 
series data and it is an improved version of the short-term memory network. The 
basic concepts in LSTM networks are “gate” and “unit” states, which are “forget 
gate,” “input gate,” “unit state,” “output gate,” and “hidden state,” respectively. The 
ci,j =
m/uni2211.s1
p=1
w i,p xti+p−1
oi =[ c1,i,c2,i,… ,cn,i]
Fig. 2  Convolutional neural network structure diagram

29
A new exploration in Baltic Dry Index forecasting learning:…
LSTM is a kind of temporal recurrent neural network (RNN). It is suitable for pro-
cessing and predicting important events with long intervals and time series delays 
and it can solve well the long-term dependence problem in simple RNNs (Fischer 
and Krauss 2018). Figure  3 shows the LSTM’s cell schematic diagram. The LSTM 
contains three gates: forget, input, and output. C t−1 is the cell state at the previous 
moment, h t−1 is the final output value of the LSTM neuron unit at the previous 
moment, xt is the input at the current moment, σ is the Sigmoid activation function, 
and tanh is also the activation function.
The LSTM calculation process is as follows:
ft is the output of the forget gate at the current moment:
it is the output of the input gate at the current moment:
̃C t is the current candidate cell state:
o t is the output value of the output gate:
C t is the cell state at the current moment:
ft = /u1D70E(W f ⋅ [ht−1 ,xt]+b f)
it = /u1D70E(W i ⋅ [ht−1 ,xt]+b i)
̃C t = tanh(WC ⋅ [ht−1,xt]+b C )
C t = ft × C t−1 + it×
∼
C t
ot = /u1D70E(W o ⋅ [ht−1 ,xt]+ bo )
Fig. 3  Long short-term memory network structure diagram

30 M. Su et al.
and h t is the output at the current moment:
Wf, Wi, WC, and Wo are the weights of the forget gate, input gate, candidate input 
gate, and output gate, respectively. In addition, bf, bi, bC, and bo are the biases of the 
forget gate, input gate, candidate input gate, and output gate, respectively. As the 
name suggests, the bi-directional long short-term memory network can memorize 
information in both directions. Its basic composition is the LSTM unit. The output 
vector of forward LSTM at time � �⃗h t and the output vector of the backward LSTM ⃖� �h t , 
after connecting left and right, becomes the output  Ht of BILSTM. Figure  4 shows 
the forward LSTM combined with the backward LSTM to form BiLSTM (Li et al. 
2018).
The output of the hidden layer of the BiLSTM cell at time t is a combination of 
the output � �⃗h t of the forward LSTM and the output ⃖� �h t of the backward LSTM:
3.2.3  Attention mechanism (AM)
The AM principle derives from the variation in human visual attention, which one 
can use to allocate different feature information resources. In recent years, combin-
ing attention models and deep learning has produced remarkable results in various 
domains like stock market predictions (Lu et al. 2021). Through the attention prob-
ability distribution, we can determine the importance of the prediction’s different 
features in the prediction problem and assign different weights to reduce the error 
between the predicted value and the actual value (Niu et al. 2021).
ht = ot × tanh(C t)
yt =[ � �⃗ht, ⃖� �ht]
Fig. 4  Bi-long short-term memory network structure diagram

31
A new exploration in Baltic Dry Index forecasting learning:…
3.3  CNN‑BiLSTM‑AM model
Figure  5 shows the CNN-BiLSTM-AM model framework. We use keras to build 
the model, and the CNN and BiLSTM layers have specific APIs in the keras 
framework, which one can directly combine and call.
The AM layer is custom-built, usually in the following three steps:
(1) Compute the similarity between the output value and input characteristic:
where W h is the weight of AM, bh is the bias of AM, and h t is the input vector, 
the BiLSTM layer’s output vector.
(2) Normalize the results in the first step, and then use the softmax function to con-
vert the attention score to obtain the attention weight coefficient:
st = tanh(W hht + bh)
Fig. 5  CNN-BiLSTM-AM model framework

32 M. Su et al.
v is the attention value, the same as the input layer vector in this study.
(3) According to the weight coefficient, the input values are weighted and summed 
to obtain the final output:
Specific information on the final model parameters can be found in Table  3 in 
Appendix.
4  Results analysis
4.1  Data processing
To use the CNN-BiLSTM-AM integrated model we need to normalize the dataset 
BDI = /braceleft.s1BDI i/uni007C.vari =1 ,2… n /braceright.s1:
In addition, because the data must enter the LSTM neural network for calculation, 
we use the first n points of BDI to forecast the BDI at the following m points. There-
fore, we must partition the dataset according to the above. As depicted in Fig.  6, the 
initial n data points predict the following m ones, generating many data pairs in suc-
cession. We then divide the dataset 7:3 to create the neural network’s training and 
test sets. Finally, we use the training set to train the model’s predictive capacity and 
the test set to evaluate the trained model’s performance.
4.2  Estimation results
We have compared in depth the prediction results of a single model versus an 
ensemble one. The result graphs of the various models follow, with the first graph 
representing the performance of the training set and the second graph illustrating the 
test set’s performance (Figs. 7, 8, 9, 10, 11, and 12).
Epochs = 1,000, Loss Function: MSE, Optimizer: Adam, Batch size: 256, Learn-
ing rate: 0.001.
4.3  Model evaluation
We employ MSE and R2 as the evaluation measures. The MSE and R 2 formulas are 
as follows:
at =
exp(sT
t v)
∑ exp(sT
t v)
s =
/uni2211.s1
t
atht
BDI
�
i = BDI i − min(BDI )
max (BDI ) − min(BDI )

33
A new exploration in Baltic Dry Index forecasting learning:…
where yi is the true value; /uni0302.s1yi is the predicted value; yi is the average value.
Following 100 simulation rounds under each model, we obtain the mean MSE 
and  R2 values for each model on the training and test sets, presented in Table  2. 
The comparison in Table  2 shows that the CNN-BiLSTM-AM integrated model 
has the best performance, with an R 2 of 96.9%.
MSE = 1
n
n/uni2211.s1
i=1
/parenleft.s1yi − /uni0302.s1yi
/parenright.s12
R 2 = 1 −
∑n
i=1
�yi − /uni0302.s1yi
�2
∑n
i=1
�
yi − /uni0302.s1yi
�2
Fig. 6  Dataset partition

34 M. Su et al.
5  Discussion and conclusion
BDI is a major indicator of the shipping sector and, often, of international trade 
and the worldwide economy (Wang et  al. 2015). In reality, forecasting BDI 
trends allows operators and investors to manage market trends and avert risks. 
This guides decision-making in other industries and manufacturers in the global 
economy (Papapostolou et al. 2016). In contrast to typical time series data, the 
BDI index exhibits volatility, non-stationarity, and complexity. Indeed, its predic-
tion is quite challenging. In light of these features of BDI, we propose an ensem-
ble model based on deep learning to forecast the trend of the BDI. We offer a 
Fig. 7  CNN model results

35
A new exploration in Baltic Dry Index forecasting learning:…
deep learning ensemble method that incorporates CNN, BiLSTM, and AM into 
a set of deep learning ensemble models, aiming to increase the accuracy of pre-
dictions. Using 6,896 observations, we have evaluated the prediction power of 
the suggested model. The study captures the hidden nonlinear features of BDI 
using machine learning techniques backed by artificial neural networks. Spe-
cifically, with CNN’s assistance, we caught hidden nonlinearities to predict BDI 
using feature data from BiLSTM and deep learning. The AM captures the effect 
of past significant temporal feature states on the BDI, to increase the accuracy of 
predictions. Our findings indicate that the CNN-BiLSTM-AM ensemble model 
outperforms a single machine learning one. Moreover, the model demonstrates 
Fig. 8  LSTM model results

36 M. Su et al.
superior performance compared to ensemble models incorporating other models 
(R2 = 96.9%).
This paper makes multiple contributions. First, the paper complements existing 
machine learning approaches in predictive research. Considering the nonlinearity 
and high volatility of the BDI, our study’s prediction findings indicate that the CNN-
BiLISM-AM deep integration model can forecast the BDI’s change trend. Deep 
learning ensemble approaches can efficiently learn meaningful information from 
features by extracting multi-layer representations of the features, hence attaining 
higher prediction accuracy than machine learning methods that use shallow features 
Fig. 9  BiLSTM model results

37
A new exploration in Baltic Dry Index forecasting learning:…
for prediction. Similar to Kamal et al.’s (2020) BDI prediction study, our results sup-
port the superior performance of LSTM-based deep ensemble models over single 
deep learning methods. In addition, compared to Makridakis et al. (2020) study, our 
model captures the variations of key nodes of the BDI and delivers more accurate 
predictions than prior studies. This finding further demonstrates the excellent imple-
mentation of AM in forecasting.
This work also contributes to shipping research. We primarily viewed the BDI 
as a forward-looking indicator of economic activity, investigated its time path, and 
presented a deep learning-based integrated model. From BDI data, we attempted to 
deduce shipping industry trends. Focusing on the value of the data has a distinct 
Fig. 10  CNN-LSTM model results

38 M. Su et al.
attraction in the age of intelligence, i.e., utilizing a data-driven model selection pro-
cedure. The technique is based entirely on the statistical features of the data, seeking 
to identify the correct univariate time series model employing heteroskedasticity, 
departure from normality, and long memory data characteristics. We concur with 
the findings of Katris and Kavussanos (2021) and extend the research on BDI uni-
variate prediction.
The study concludes by offering certain management insights. Our Mari-
time Transport Forecast Model serves as a valuable tool in providing insights to 
the global economic landscape. Thus, we can deduce economic activity trends 
from shipping statistics. First, shipowners can make business decisions using 
Fig. 11  CNN-BiLSTM model results

39
A new exploration in Baltic Dry Index forecasting learning:…
Fig. 12  CNN-BiLSTM-AM model results
Table 2  Model performance comparison
Model Training set Test set Training set R2 Test set R2
CNN 8.7146e−04 0.0011 97.3 81.8
LSTM 2.6141e−04 3.3773e−04 99.1 94.6
BiLSTM 2.3394e−04 3.4468e−04 99.2 94.3
CNN-LSTM 2.7196e−04 3.2752e−04 99.1 94.7
CNN-BiLSTM 2.9995e−04 3.2857e−04 99.0 94.6
CNN-BiLSTM-AM 1.4476e−04 2.1469e−04 99.7 96.9

40 M. Su et al.
short-term information about seasonal swings in the bulk cargo industry. If the 
expected BDI is low, the ship can go to a dry dock for maintenance and repairs 
or be laid-up or sold. When the BDI is expected to be high, shipowners, often 
within alliances, can coordinate better operations and investments for profit opti-
mization. When the BDI trend rises, shipping corporations can decide to increase 
the speed of their ships to boost productivity. Moreover, investors can consider 
purchasing, holding, or selling shipping company stocks when considering the 
anticipated volatility of the BDI.
This study is not void of drawbacks. First, it is suggested that focusing on the 
value of BDI data itself may be an intriguing topic. However, the study focuses 
solely on the changes in the BDI time series, a single dataset. Second, we com-
bine a limited number of machine learning models into model ensembles. As a 
result, it is easy to overlook more combinations of machine learning models with 
better performance. Third, the duration predictions may not be optimal. Future 
research models should enhance prediction lengths. Therefore, we advise includ-
ing more market-influencing factors and carrying out multi-factor projection 
research. In addition, we propose discovering more efficient deep learning tech-
niques to improve prediction cycle and accuracy.
Appendix
Table 3 illustrates the specific information of the final model parameters.
Table 3  Final model parameters Layer (Type) Output shape Param#
Input_1 (InputLayer) [(None, 14, 1)] 0
Conv1d (conv1D) (None, 14, 64) 256
Dropout (Dropout) (None, 14, 64) 0
Bi-directional (Bi-directional) (None, 14, 128) 66,048
Dropout_1 (Dropout) (None, 14, 128) 0
Dense (Dense) (None, 14, 128) 16,512
Attention_vec (Permute) (None, 14, 128) 0
Multiply (Multiply) (None, 14, 128) 0
Flatten (Flatten) (None, 14, 128) 0
Dense_1 (Dense) (None, 1) 1793
Total params: 84,609
Trainable params: 84,609
Non-trainable params: 0

41
A new exploration in Baltic Dry Index forecasting learning:…
Acknowledgements We appreciate the comments and opinions of MEL editors and reviewers; they were 
highly insightful and beneficial in updating and refining our work, as well as providing essential guidance 
for our research.
References
Alizadeh, Amir H., and Gulnur Muradoglu. 2014. Stock market efficiency and international shipping-
market information. Journal of International Financial Markets, Institutions and Money 33: 
445–461.
Apergis, Nicholas, and James E. Payne. 2013. New evidence on the information and predictive content of 
the Baltic Dry Index. International Journal of Financial Studies 1 (3): 62–80.
Bae, Sung-Hoon., Gunwoo Lee, and Keun-Sik. Park. 2021. A Baltic Dry Index prediction using deep 
learning models. Journal of Korea Trade 25 (4): 17–36.
Bandyopadhyay, Arunava, and Prabina Rajib. 2023. The asymmetric relationship between Baltic Dry 
Index and commodity spot prices: Evidence from nonparametric causality-in-quantiles test. Mineral 
Economics 36 (2): 217–237.
Batchelor, Roy, Amir Alizadeh, and Ilias Visvikis. 2007. Forecasting spot and forward prices in the inter-
national freight market. International Journal of Forecasting 23 (1): 101–114.
Bildirici, Melike E., Fazıl Kayıkçı, and I.şıl Şahin Onat. 2015. Baltic Dry Index as a major economic 
policy indicator: The relationship with economic growth. Procedia - Social and Behavioral Sciences 
210: 416–424.
Chang, Chao-Chi., Heng Chih Chou, and Wu. Chun Chou. 2014. Value-at-risk analysis of the asymmet-
ric long-memory volatility process of dry bulk freight rates. Maritime Economics & Logistics 16 
(3): 298–320.
Chen, Shun, Hilde Meersman, and Eddy van de Voorde. 2012. Forecasting spot rates at main routes in the 
dry bulk market. Maritime Economics & Logistics 14 (4): 498–537.
Chou, Chien-Chang., and Keng-Shou. Lin. 2019. A fuzzy neural network combined with technical indi-
cators and its application to Baltic Dry Index forecasting. Journal of Marine Engineering & Tech-
nology 18 (2): 82–91.
Cromwell, Jeff B., Michael J. Hannan, Walter C. Labys, and Michel Terraza. 1994. Multivariate tests for 
time series models. Thousand Oaks: SAGE Publications.
Cullinane, Kevin, and Hercules Haralambides. 2021. Global trends in maritime and port economics: The 
COVID-19 pandemic and beyond. Maritime Economics & Logistics 23 (3): 369–380.
Cullinane, Kevin, Keith Mason, and Matthew Cape. 1999. A comparison of models for forecasting the 
Baltic freight index: Box-Jenkins revisited. International Journal of Maritime Economics 1 (2): 
15–39.
Duru, Okan, and Shigeru Yoshida. 2009. Judgmental forecasting in the dry bulk shipping business: Sta-
tistical vs. judgmental approach. Asian Journal of Shipping and Logistics 25 (2): 189–217.
Duru, Okan, Emrah Bulut, and Shigeru Yoshida. 2012. A fuzzy extended DELPHI method for adjustment 
of statistical time series prediction: An empirical study on dry bulk freight market case. Expert Sys-
tems with Applications 39 (1): 840–848.
Duru, Okan. 2010. A fuzzy integrated logical forecasting model for dry bulk shipping index forecasting: 
An improved fuzzy time series approach. Expert Systems with Applications 37 (7): 5372–5380.
Fischer, Thomas, and Christopher Krauss. 2018. Deep learning with long short-term memory networks 
for financial market predictions. European Journal of Operational Research 270 (2): 654–669.
Fuller, Wayne A. 2009. Introduction to statistical time series. Hoboken: John Wiley & Sons.
Gao, Ruzhao, Yueqiang Zhao, and Bing Zhang. 2023. Baltic Dry Index and global economic policy 
uncertainty: Evidence from the linear and nonlinear Granger causality tests. Applied Economics Let-
ters 30 (3): 360–366.
Gu, Jiuxiang, Zhenhua Wang, Jason Kuen, Lianyang Ma, Amir Shahroudy, Bing Shuai, Ting Liu, Xingx-
ing Wang, Gang Wang, Jianfei Cai, and Tsuhan Chen. 2018. Recent advances in convolutional neu-
ral networks. Pattern Recognition 77 (C): 354–377.
Gu, Yimiao, Zhenxi Chen, and Donald Lien. 2019. Baltic Dry Index and iron ore spot market: Dynamics 
and interactions. Applied Economics 51 (35): 3855–3863.

42 M. Su et al.
Guan, Feng, Zixuan Peng, Keming Wang, Xiaolin Song, and Junjie Gao. 2016. Multi-step hybrid predic-
tion model of Baltic Supermax Index based on support vector machine. Neural Network World 26 
(3): 219–232.
Han, Liyan, Li. Wan, and Xu. Yang. 2020. Can the Baltic Dry Index predict foreign exchange rates? 
Finance Research Letters 32 (C): 101157.
Han, Qianqian, Bo. Yan, Guobao Ning, and B. Yu. 2014. Forecasting dry bulk freight index with 
improved SVM. Mathematical Problems in Engineering 2014: 460684.
Haralambides, Hercules E. 2019. Gigantism in container shipping, ports and global logistics: A time-
lapse into the future. Maritime Economics & Logistics 21 (1): 1–60.
Kamal, Imam Mustafa, Hyerim Bae, Sim Sunghyun, and Heesung Yun. 2020. DERN: Deep ensemble 
learning model for short- and long-term prediction of Baltic Dry Index. Applied Sciences 10 (4): 
1504.
Kanamoto, Kei, Liwen Murong, Minato Nakashima, and Ryuichi Shibasaki. 2021. Can maritime big 
data be applied to shipping industry analysis? Focussing on commodities and vessel sizes of dry 
bulk carriers. Maritime Economics & Logistics 23 (2): 211–236.
Katris, Christos, and Manolis G. Kavussanos. 2021. Time series forecasting methods for the Baltic 
Dry Index. Journal of Forecasting 40 (8): 1540–1565.
Kavussanos, Manolis G., and Amir H. Alizadeh-M. 2001. Seasonality patterns in dry bulk shipping 
spot and time charter freight rates. Transportation Research Part e: Logistics and Transportation 
Review 37 (6): 443–467.
Leonov, Yordan, and Ventsislav Nikolov. 2012. A wavelet and neural network model for the prediction 
of dry bulk shipping indices. Maritime Economics & Logistics 14 (3): 319–333.
Li, Fei, Meishan Zhang, Bo. Tian, Bo. Chen, Fu. Guohong, and Donghong Ji. 2018. Recognizing 
irregular entities in biomedical text via deep neural networks. Pattern Recognition Letters 105: 
105–113.
Lin, Arthur J., Hai Yen Chang, and Jung Lieh Hsiao. 2019. Does the Baltic Dry Index drive volatil-
ity spillovers in the commodities, currency, or stock markets? Transportation Research Part e: 
Logistics and Transportation Review 127 (C): 265–283.
Lin, Faqin, and Nicholas C.S.. Sim. 2013. Trade, income and the Baltic Dry Index. European Eco-
nomic Review 59 (C): 1–18.
Lin, Hsio-Yi, Yu-Fang Juan, and An-Pin Chen. 2007. Hybrid intelligent trading approach XCS neural 
network model for Taiwan Stock index trend forecasting. In Proceedings of the 2007 Interna-
tional Conference on Convergence Information Technology, November 21–23, 2007, Gwangju, 
Korea, 1408–1416. Washington, DC: IEEE Computer Society.
Liu, Mingxi, Yajie Zhao, Jingkai Wang, Chang Liu, and Guowen Li. 2022. A deep learning frame-
work for Baltic Dry Index forecasting. Procedia Computer Science 199: 821–828.
Ljung, Greta M., and George E.P.. Box. 1978. On a measure of lack of fit in time series models. Biom-
etrika 65 (2): 297–303.
Lu, Wenjie, Jiazheng Li, Jingyang Wang, and Lele Qin. 2021. A CNN-BiLSTM-AM method for stock 
price prediction. Neural Computing and Applications 33 (10): 4741–4753.
Makridakis, Spyros, et al. 2020. A novel forecasting model for the Baltic dry index utilizing optimal 
squeezing. Journal of Forecasting 39 (1): 56–68.
Niu, Zhaoyang, Guoqiang Zhong, and Yu. Hui. 2021. A review on the attention mechanism of deep 
learning. Neurocomputing 452: 48–62.
Notteboom, Theo E., and Hercules E. Haralambides. 2020. Port management and governance in a 
post-COVID-19 era: Quo vadis? Maritime Economics & Logistics 22 (3): 329–352.
Oomen, J.G.M. 2012. The Baltic Dry Index: A predictor of stock market returns. Master’s thesis, Til-
burg University.
Papailias, Fotis, Dimitrios D. Thomakos, and Jiadong Liu. 2017. The Baltic Dry Index: Cyclicalities, 
forecasting and hedging strategies. Empirical Economics 52 (1): 255–282.
Papapostolou, Nikos C., Panos K. Pouliasis, Nikos K. Nomikos, and Ioannis Kyriakou. 2016. Ship-
ping investor sentiment and international stock return predictability. Transportation Research 
Part E: Logistics and Transportation Review 96: 81–84.
Şahin, Bekir, Samet Gürgen, Bedir Ünver, and İsmail Altin. 2018. Forecasting the Baltic Dry Index by 
using an artificial neural network approach. Turkish Journal of Electrical Engineering & Com-
puter Sciences 26 (3): 1673–1684.

43
A new exploration in Baltic Dry Index forecasting learning:…
Said, Husaini, and Evangelos Giouvris. 2019. Oil, the Baltic Dry index, market (il)liquidity and busi-
ness cycles: Evidence from net oil-exporting/oil-importing countries. Financial Markets and 
Portfolio Management 33 (4): 349–416.
Tsioumas, Vangelis, Stratos Papadimitriou, Yiannis Smirlis, and Shaher Zahran Zahran. 2017. A 
novel approach to forecasting the bulk freight market. Asian Journal of Shipping and Logistics  
33 (1): 33–41.
Uyar, Kaan, Ümit. ilhan, and Ahmet İlhan. 2016. Long term dry cargo freight rates forecasting by 
using recurrent fuzzy neural networks. Procedia Computer Science 102 (C): 642–647.
Veenstra, Albert Willem, and Philip Hans Franses. 1997. A co-integration approach to forecasting freight 
rates in the dry bulk shipping sector. Transportation Research Part a: Policy and Practice 31 (6): 
447–458.
Wang, Yadong, Qiang Meng, and Du. Yuquan. 2015. Liner container seasonal shipping revenue manage-
ment. Transportation Research Part b: Methodological 82 (C): 141–161.
Xiao, Wei, Xu. Chuan, Hongling Liu, and Xiaobo Liu. 2021. A hybrid LSTM-Based ensemble learning 
approach for China coastal bulk coal freight index prediction. Journal of Advanced Transportation 
2021: 5573650.
Yang, Hualong, Fang Dong, and Margarette Ogandaga. 2008. Forewarning of freight rate in shipping 
market based on support vector machine. In Traffic and transportation studies, ed. Baohua Mao, 
Zongzhong Tian, Haijun Huang, and Ziyou Gao, 295–303. Reston: American Society of Civil 
Engineers.
Yang, Zaili, and Esin Erol Mehmed. 2019. Artificial neural networks in freight rate forecasting. Maritime 
Economics & Logistics 21 (3): 390–414.
Zeng, Qingcheng, Qu. Chenrui, Adolf K.Y.. Ng, and Xiaofeng Zhao. 2016. A new approach for Baltic 
Dry Index forecasting based on empirical mode decomposition and neural networks. Maritime Eco-
nomics & Logistics 18 (2): 192–210.
Zhang, Xin, M.Y. Chen, Minggang Wang, Ying-en Ge, and H. Eugene Stanley. 2019. A novel hybrid 
approach to Baltic Dry Index forecasting based on a combined dynamic fluctuation network and 
artificial intelligence method. Applied Mathematics and Computation 361 (C): 499–516.
Zhang, Xin, Tianyuan Xue, and H. Eugene Stanley. 2018. Comparison of econometric models and arti-
ficial neural networks algorithms for the prediction of baltic dry index. IEEE Access 7: 1647–1657.
Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps 
and institutional affiliations.
Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under 
a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted 
manuscript version of this article is solely governed by the terms of such publishing agreement and 
applicable law.