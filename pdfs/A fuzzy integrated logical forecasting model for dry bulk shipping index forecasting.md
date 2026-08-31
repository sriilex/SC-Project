# A fuzzy integrated logical forecasting model for dry bulk shipping index forecasting: An improved fuzzy time series approach

Okan Duru *
Department of Maritime Transportation and Management Engineering, Istanbul Technical University, Tuzla 34940, Istanbul, Turkey

## ARTICLE INFO
Keywords:
Fuzzy time series
Linguistic variable
Forecasting
Shipping index

## ABSTRACT
This study develops an improved fuzzy time series method via adjustment of the latest value factor and previous error patterns. There are many fuzzy extended applications in the literature, and the fuzzy time series is one successful implementation of fuzzy logical modelling. Fuzzy time series have been studied for over a decade, and many researchers have proposed to remove some of the drawbacks of the initial fuzzy time series algorithm. In this paper, fuzzy integrated logical forecasting (FILF) and extended FILF (E-FILF) algorithms are suggested for short term forecasting purposes. Empirical studies are performed over the Baltic Dry Index (BDI), and indicate the superiority of the proposed approach compared to conventional benchmark methods.
2010 Elsevier Ltd. All rights reserved.

## 1. Introduction
Forecasting tasks are a crucial activity in all business types, and are necessary for planning and developing strategies even if the forecasting results are inferior. In business exercises, time series techniques are the most applied methodology for prediction objectives. A variety of seminal papers have suggested improvements to the prediction accuracy using techniques such as moving averages, auto regression, and smoothing methods (Holt, 1957; Winters, 1960; Box & Jenkins, 1976; Bowerman & O’Connell, 1979; Harvey, 1990). Most of these methods are able to predict a particular type of data accurately, but they are inappropriate for many time series in practical life. An econometric model, which is based on casual relationships, requires normality and stationarity of the data, as well as a large data set. The conventional time series extrapolation requires normality and stationarity (constant mean and constant variance) as well. However, many data sets are not stationary, and special care must be taken to implement methods and extend recent problems.

After the development of the fuzzy set theory (FST), a new generation of time series methods has been implemented using the fuzzy time series (FTS) approach (Song & Chissom, 1993a; Zadeh, 1965). The FTS method does not require a large data sample, stationarity, normality, or a purely quantitative data set. The FTS can operate on linguistic variables, and the traditional fuzzification of a time series is a transformation of quantitative data to linguistic terms. It is generally based on data consolidation into intervals by a specified procedure (Palit & Popovic, 2005).

The FTS has been developed and implemented in various studies (Chen, 1996; Chen & Hwang, 2000; Cheng, Chen, Teoh, & Chiang, 2008; Huarng, 2001; Huarng & Yu, 2005, 2006; Hwang, Chen, & Lee, 1998; Liu, 2007; Song & Chissom, 1993a, 1993b, 1994; Sullivan & Woodall, 1994; Yu, 2005). Song and Chissom (1993a, 1993b) first showed the application of the FST to analysis and forecasting of a time series. Later, Chen (1996) developed an initial study that improved the arithmetic operations rather than the logic max-min composition methodology of Song and Chissom. This method also provides robust predictions when the historical data are not accurate for a forecasting task. Huarng (2001) consolidated the study of Chen with his heuristic rule structure. Yu (2005) suggested a weighting algorithm for fuzzy logical relationships (FLRs). This study improved upon previous results by showing that highly probable movements have a larger effect on FLRs. The weighting algorithm can be performed by an expert judgment or the latest FLR weighting approach, or it can be calculated from the existence density of FLRs. Liu (2007) extended previous work with the trapezoidal design of the FTS. Chu, Chen, Cheng, and Huang (2009) developed a model to implement the causality of various time-series using a fuzzy dual-time series algorithm. Their paper introduced a dual-factor approach to a forecasting task using TAIEX (Taiwan stock exchange capitalization weighted stock index) and NASDAQ (National association of securities dealers automated quotations) index data, and used the dynamics of stock markets based on price–volume relationships.

The present paper suggests an improved FTS model, the FILF (Fuzzy Integrated Logical Forecast Model), which reduces model errors by the latest value adjustment algorithm. The FILF methodology also proposes an error correction function that manages the last error rates and the pattern of error series, which is called E-FILF (extended FILF).

[Figure 1: Baltic dry index (BDI raw data) - Line chart]

Empirical studies are conducted with data from an international shipping freight index, BDI, which is specified by the Baltic Exchange, London. The price of maritime transportation is expressed as freight rate, and the BDI is a unique indicator of world shipping freights. It is a combined price for the main shipping routes and contracts. Forecasting of shipping freights is of great importance to ship management companies, but it is also crucial for charterers, who are made up of the industries and manufacturers of the world economic system. The shipping freights represent a considerable proportion of the price of finished goods, so shipping freights and wholesale prices have a strong relationship that must be predicted for the business and financial planning of many stakeholders in the economy (Metaxas, 1971). Fig. 1 shows the series of the BDI, which consists of data from January 2001 to November 2008 (monthly data, which is the simple average of the daily indices).

Forecasting the shipping freight market has been attempted in a series of studies, most based on econometric modelling, simultaneous equations models, and time-series analysis methods (Beenstock & Vergottis, 1993; Charemza & Gronicki, 1981; Cullinane, 1992; Glen, 1997; Hale & Vanags, 1992; Hampton, 1991; Hawdon, 1978; Kavussanos, 1996; Kavussanos, 1997; Shimojo, 1979; Tinbergen, 1959). Tinbergen (1959) first described the long-term dynamics of shipping markets. In the last half century, many researchers have tried to model and forecast shipping markets. However, the restrictions of statistical extrapolation lead to many inconsistent results when using traditional methods. The cycles and behaviour of the markets change every day. Furthermore, an alternative to time-series methodology is the judgmental forecasting that is also applied in the shipping practice (Goodwin & Wright, 1993; Sanders, 1992). Delphi-based and expert opinion-based studies are performed to increase the accuracy of predictions, and to understand the judgmental dynamics of the markets (Ariel, 1989; Duru & Yoshida, 2008a, 2008b, 2009). Duru and Yoshida (2008a, 2008b) suggested the development of a hybrid model of forecasting methodology to generate consensus results in freight market forecasting.

This paper is organized as follows: Section 2 reviews conventional FTS analysis, and presents the FILF and E-FILF algorithms. Section 3 describes the procedure for the applications in detail with examples. Section 4 discusses the results and accuracy of the proposed models relative to benchmark methods. Section 5 concludes the present paper.

## 2. Methodology

### 2.1. Fuzzy time series

Song and Chissom (1993a, 1993b) proposed the FTS to model fuzzy logical relationships (FLRs) among data. A fuzzy set is a group of data that has a grade of membership through the mentioned fuzzy set. Let U be the universe of discourse with U =(u1, u2, ... , um) where ui are linguistic variables. The basic definitions of FTS are as follows:

Definition 1. Y(t)(t = ..., 0, 1, 2, ...) is a subset of real numbers. Let Y(t) be the universe of discourse defined by the fuzzy set μi(t). If F(t) consists of μi(t)(i = 1, 2, ...), F(t) is called a fuzzy time series on Y(t).

Definition 2. If there exists a fuzzy relationship R(t - 1, t) such that F(t)= F(t - 1) ° R(t - 1, t), where ° is an arithmetic operator, then F(t) is said to be caused by F(t - 1). The relationship between F(t) and F(t - 1) can be denoted by F(t - 1) -> F(t).

Definition 3. Suppose F(t) is calculated by F(t - 1) only, and F(t)= F(t - 1) ° R(t - 1, t). For any t, if R(t - 1, t) is independent of t, then F(t) is considered a time-invariant fuzzy time series. Otherwise, F(t) is time-variant.

Definition 4. Suppose F(t - 1) = Ãi and F(t) = Ãj, a fuzzy logical relationship can be defined as Ãi -> Ãj, where Ãi and Ãj are called the left-hand side (LHS) and right-hand side (RHS) of the FLR, respectively.

Chen (1996) developed the method of Song and Chissom to ensure more accurate results. As a benchmark of the proposed model, the procedure of Chen’s methodology is as follows:

Step 1. Divide the universe of discourse U into equal-length intervals.
Step 2. Define the fuzzy sets on U, fuzzify the historical data, and derive the FLRs.
Step 3. Allocate the derived fuzzy logical relationships into groups.
Step 4. Calculate the forecasted values under the three defuzzification rules.

### 2.2. Fuzzy integrated logical forecasting model (FILF)

A differencing operation provides a modified series from the original data using its first order or higher order differences. In the traditional time series analysis, the differencing method is suggested to satisfy the stationarity restriction (Box & Jenkins, 1976; Newbold, 1975). Highly volatile data sets in particular are transformed by differencing to maintain a tight data range. Differencing reduces the standard deviation of the data, and also reduces the length of intervals. One of the drawbacks of the FTS originates from volatile data forecasting. The FTS builds FLRs from the data, and if the data is highly volatile, the RHS of the FLRs can be highly volatile as well. For instance, defuzzification results can be a centroid of two extreme fuzzy sets.

Lemma. Let Y(t) be a subset of real numbers and defined by the universe of discourse U that consists of the fuzzy sets Ãi(i = 1, 2, ...). If the data set Y(t) stems from a highly volatile system, the result of the FLRs can be an overestimation or an underestimation of the actual value with excessive error.

Proof. Let ÃO be an objective value in the LHS of a FLRs group, ÃU is an upper extreme fuzzy set, ÃL is a lower extreme fuzzy set, and the FLRs group is defined as ÃO -> ÃU, ÃL. If ÃO is a fuzzy set near the lower bound, then the forecast of the fuzzy algorithm may be an overestimate at all iterations. Likewise, if ÃO is a fuzzy set near the upper bound, then the forecast of the fuzzy algorithm may be an underestimate at all iterations. 

Remark. Level-based fuzzy time series models (no differencing, or no trend modelling) are subject to the above-mentioned errors, and also cannot sort the direction of one-period-ahead values. A differencing operation ensures the processing of different directions (upward trend or downward trend), and also handles the errors originating from high volatility.

The second treatment of the FILF algorithm is the latest value adjustment. Although, FLRs provide a logical movement of historical data, the post-sample performance is subject to conversation of the current logical system. However, the practical time series data have many unusual movements, and the recent data can be sourced from a different logical algorithm (Goodwin & Fildes, 1999). To increase the post-sample accuracy of the fuzzy time series model, the FILF algorithm uses the latest value adjustment function. Definitions for the improved model are as follows:

Definition 5. The lag, or a backward linear function for raw data that defines the first order differences of the original series, is as follows:
ΔY(t) = Y(t) - Y(t - 1) (1)

Definition 6. β is an adjustment coefficient that defines the combination function of the last actual value of the fuzzified data set and the forecasted value for t + 1. The fuzzified data can be the raw time series data, the first differenced data or the second differenced set as well.
FR(t + 1) = Y(t) * β + F(t + 1)(1 - β) (2)
β ∈ [0, 1]

Property. The adjustment coefficient β can be defined by experimental studies, and can also be calculated by a simulation of the function to minimize errors in the estimation period of the data.

Definition 7. A FILF algorithm is described by its order: FILF (i, d, β)
i: number of fuzzy sets
d: order of differencing operator (Δd Y(t))
β: value of adjustment coefficient

Example 1. If the FILF algorithm is specified with seven fuzzy numbers (Ãi, i = 1, 2, ... , 7), the first order differenced series (d = 1), and the adjustment coefficient is 0.6 (β = 0.6), then the specification is FILF (7,1,0.6).

#### 2.2.1. Program 1. The FILF procedure
Step 1. Define the universe of discourse U. If the original data is differenced, the differenced data will be defined by the universe of discourse U.
Step 2. Divide U into intervals according to linguistic terms.
Step 3. Define the fuzzy sets on U, and fuzzify the historical data.
Step 4. Derive the FLRs based on the historical data.
Step 5. Classify the derived FLRs into groups.
Step 6. Utilize three defuzzification rules to calculate the forecasted values.
Step 7. Regulate the forecasted values by the combination function of the latest actual value of fuzzified data set and forecasted value.

### 2.3. Extended fuzzy integrated logical forecasting model (E-FILF)

A prediction task can be evaluated by its errors from the actual system. Most of the studies conduct an error measurement method such as mean absolute percentage error (MAPE), mean squared error (MSE), or root mean squared error (RMSE). The traditional procedure is to compare the proposed model with benchmark methods. However, we know that one unique piece of information about the model is the pattern of errors. In the FTS literature, it is not common that errors are investigated whether or not there is evidence of a consistent pattern. Nevertheless, the classical econometric analysis aims to exclude correlated errors that consist of signs of some other factors that cannot defined by the model. An error correction function can improve the accuracy of a fuzzy time series model when a pattern or correlation of errors exists.

Definition 8. A percentage error (PE) is defined by
PE = (Dvt - Fvt) / Dvt (3)
Dvt: actual value of time t
Fvt: forecasted value of time t

and an absolute percentage error (APE) is as follows
APE = |(Dvt - Fvt) / Dvt| (4)
Dvt: actual value of time t
Fvt: forecasted value of time t

[Figure 2: The PEs of the FILF model for the BDI index - Line chart]

Fig. 2 shows the PEs of BDI forecasting after the FILF results. The average of the PEs is -0.01. In our prediction task, a minor deviation exists from the standard white noise error rates. An error correction of the forecasted data can improve the accuracy of the model. If we conduct the same procedure on the Naïve forecasts of the BDI (the forecasted value of time t equals to the value of time t - 1) , an error correction procedure reduces MAPE from 0.12 to 0.09.
The classical time series analysis investigates the model errors, and the series of errors is expected to be white noise that denotes a normally, identically, independently distributed variable (Makridakis, Wheelwright, & Hyndman, 1998). The treatment of an error series that is not white noise is modelled by an error correction algorithm. The correction is processed by a simple moving average (SMA) of previous error rates.

Definition 9. An error correction function is defined as follows:
A simple moving average (SMA) is the unweighted mean (simple average) of the previous q data points. For error correction, an SMA of percentage errors of the model is calculated as
SMAε = (εt + εt-1 + ... + εt-q) / q (5)
q is an integer number that denotes the SMAε horizon for previous errors. The corrected forecast, FC(t + 1), is
FC(t + 1) = FR(t + 1) + FR(t + 1) * SMAε (6)

Definition 10. An error correction modified FILF is defined as an E-FILF model that has specification.
E-FILF (i, d, β, q)
i: number of fuzzy sets
d: order of differencing operator
β: value of adjustment coefficient
q: the SMAε horizon for previous errors

Example 2. If the E-FILF algorithm is specified with 5 fuzzy numbers (Ãi, i = 1, 2, ... , 5), the first order differenced series (d = 1), the adjustment coefficient is 0.4 (β = 0.4), and the SMAε horizon is 6 period backward, then the specification is E-FILF (5,1,0.4,6).

#### 2.4.1. Program 2. The E-FILF procedure
Step 1-7. As defined in the FILF procedure.
Step 8. Employ the error correction function within a backward horizon, which is user-defined, or a possible range of moving average terms will be simulated for error minimization.

[Figure 3: The process of FILF and E-FILF algorithms - Flowchart]

## 3. Application of proposed models

The current fuzzy time series models (Chen, 1996, 2002; Chen & Hwang, 2000; Huarng, 2001; Hwang et al., 1998; Lee & Chou, 2004; Song & Chissom, 1993a, 1993b, 1994) utilize discrete fuzzy sets to define their fuzzy time series. Their discrete fuzzy sets are defined as follows:
Assume there are m intervals, which are u1 = [d1,d2], u2 = [d2,d3], u3 = [d3,d4], u4 = [d4,d5], ... , um-3 = [dm-3,dm-2], um-2 = [dm-2,dm-1], um-1 = [dm-1, dm], and um = [dm,dm+1].
Let Ã1, Ã2, ... , Ãk be fuzzy sets that are linguistic values of the data set. Define fuzzy sets Ã1, Ã2, ... , Ãk on the universe of discourse U as follows:
Ã1 = a11/u1 + a12/u2 + a13/u3 + ... + a1m/um,
Ã2 = a21/u1 + a22/u2 + a23/u3 + ... + a2m/um,
...
Ãk = ak1/u1 + ak2/u2 + ak3/u3 + ... + akm/um,
where aij ∈ [0,1], 1 <= i <= k, and 1 <= i <= m. The value of aij indicates the grade of membership of uj in the fuzzy set Ãi. The degree of each data is determined according to their membership grade to the fuzzy sets. When the maximum membership grade exists in Ãk, the fuzzified data is treated as Ãk. Our empirical study about dry bulk shipping index is defined by the following linguistic fuzzy sets: Ã1(very very few), Ã2(very few), Ã3(few), Ã4 (moderate), Ã5 (more), Ã6 (very more) and Ã7 (very very more).

[Table 1: First order differences of the BDI data (sample data).]
[Table 2: Descriptive statistics of the BDI raw dataset and the 1st order differences.]

Define the universe of discourse U. Find the maximum Dmax and the minimum Dmin among all Dvt. For easy portioning of U, two small numbers D1 and D2 are assigned. The universe of discourse U is then defined by: U = [Dmin - D1, Dmax + D2]
Table 2 shows the descriptive statistics of the raw BDI data and its 1st order differenced version. As we conclude from the information, the first order differencing provided to reduce the data range, and decrease the standard deviation. The universe of discourse U is defined as:
U = [-3167.57 - 32.43, 2556.79 + 43.21] = [-3200, 2600]
D1 and D2 are 32.43 and 43.21, respectively.

Step 1. Determine the length of the interval l according to linguistic variables. Seven linguistic variables are defined for this study. For increasing sensitivity, the largest density interval is divided into four sub-intervals, and the second largest density interval is divided into three sub-intervals, as suggested by Wang and Hsu (2008). For these purposes, the densities of all intervals are calculated, and then the two largest intervals are processed according to the sub-division procedure. Table 3 presents assigned fuzzy sets and their specifications including intervals, midpoints and linguistic terms.

There are seven intervals and sub-intervals. Seven major fuzzy intervals are u1 = [-3200, -2371.43], u2 = [-2371.43, -1542.86], u3 = [-1542.86, -714.29], u4 = [-714.29, 114.29], u5 = [114.29, 942.86], u6 = [942.86, 1771.43] and u7 = [1771.43, 2600]. Subintervals are u41 = [-714.29, -507.1], u42 = [-507.1, -300.0], u43 = [-300.0, -92.9], u44 = [-92.9, 114.29], u51 = [114.29, 390.5], u52 = [390.5, 666.7] and u53 = [666.7, 942.86]. Total number of intervals is 14, so the specification of the model is FILF (14,1, β) in step 2.

Step 1. Fuzzification of the first order difference of BDI is performed and presented as in Table 4. The data is associated with fuzzy sets according to their maximum membership grade in a fuzzy set.

[Table 3: The fuzzy sets and their specifications.]
[Table 4: The fuzzified values of the first order differenced BDI series.]

Step 2. Generate the FLRs. For all fuzzified data, derive the FLRs according to Definition 4 such as... Ã52 -> Ã53, Ã53 -> Ã51, Ã51 -> Ã3,... (from Table 4).
Step 3. Organize the FLRs into groups of the same LHS fuzzy sets named the FLR Group (FLRG). The LHSs of the groups indicate the input value, which is the 1st order differencing of one period of previous data. The RHS is the variety of outputs that occurred in the estimation period. Table 5 shows FLRGs.

[Table 5: List of the FLRGs of the first order differenced BDI series.]

Step 4. Calculate the forecasted outputs. The forecasted value at time t, Fvt, is determined by the following three IF-THEN rules. Assume the fuzzy number of Dvt-1 at time t - 1 is Ãj.

Rule 1. IF the FLRG of Ãj does not exist; Ãj -> φ, THEN the value of Fvt is Ãj, and we calculate the centroid of the fuzzy set Ãj, which is located on midpoint, for inferring the point forecast.

Rule 2. IF the FLRG of Ãj is one-to-one; Ãj -> Ãk, THEN the value of Fvt is Ãk, and we calculate the centroid of the fuzzy set Ãk, which is located on the midpoint, for inferring the point forecast.

Rule 3. IF the FLRG of Ãj is one-to-many; Ãj -> Ãk1, Ãj -> Ãk2, Ãj -> Ãk3, ... , Ãj -> Ãkp, THEN the value of Fvt is calculated as follows:
Fvt = Ãk1 + Ãk2 + ... + Ãkp / p (8)
and we calculate the centroid of the resulting fuzzy set, which is the arithmetic average of mk1, mk2, ... , mkp, the midpoints of uk1, uk2, ... , ukp, respectively.

[Figure 4: Comparative chart of the beta coefficient and MAPE for the forecasted data.]

The fuzzy sets Ã1, Ã2, ... , Ãk are defined by
Ã1 = 1/u1 + 0.5/u2 + 0/u3 + 0/u4 + ... + 0/um,
Ã2 = 0.5/u1 + 1/u2 + 0.5/u3 + 0/u4 + ... + 0/um,
Ã3 = 0/u1 + 0.5/u2 + 1/u3 + 0.5/u4 + ... + 0/um,
...
Ãk-1 = 0/u1 + 0/u2 + ... + 0/um-3 + 0.5/um-2 + 1/um-1 + 0.5/um,
Ãk = 0/u1 + 0/u2 + ... + 0/um-3 + 0/um-2 + 0.5/um-1 + 1/um,

For a performance evaluation of the proposed model, mean absolute percentage error (MAPE) results will be compared to benchmark methods. The MAPE is calculated as follows:
MAPE = 1/n * Σ |(Dvt - Fvt) / Dvt| (7)
Dvt: actual value of time t
Fvt: forecasted value of time t

The detailed application steps of the FILF and E-FILF can be described as follows:
...
Example. The raw values for January and February 2006 are 2261.76 and 2443.70 respectively. The first order difference is 181.94 for February 2006, which has a maximum membership grade in the fuzzy set Ã51. The result of the FLRG of Ã51 is +221.7. Before the adjustment process, the forecasted value of March 2006 is DvMAR06 = DvFEB06 + 221.7 = 2443.70 + 221.7 = 2665.4 (actual value of BDIMAR06 is 2598.83).

Step 7. Establish the adjustment process. In the adjustment process, the determination of the β coefficient can be performed with a simulation to minimize errors, or the user can use a predetermined value. If the β coefficient is near 1.0, then the process provides the latest-value sensitive results. If the β coefficient is near 0.0, then process provides the forecasted-value sensitive results.

For the forecasting task of BDI, a simulation of the β coefficient is conducted with respect to minimization of the MAPE. Fig. 4 shows the result of this simulation, and the β coefficient that minimized the MAPE was 0.68. Therefore, the model chooses FILF (14,1,0.68). The result of this process gives the regulated forecast FR(t + 1).

[Figure 5: The PEs of the E-FILF (14,1,0.68,13) model for BDI.]
[Table 6: Overall results of fuzzy forecasting methods.]
[Figure 6: The absolute percentage errors of the methods.]

Step 7 (continued). The process of error correction is not necessary for all time series, but it is suggested particularly if the data denotes a particular pattern of the errors. The determination of the SMAε is performed by minimisation of the MAPE, or the user himself defines it. If the backward horizon q is increased, that will produce a model with a long memory. Otherwise, if the backward horizon is closer, the model will have a short memory. As a result of error correction, the corrected forecast FC(t + 1) is indicated.

The E-FILF (14,1,0.68,13) algorithm extends the FILF (14,1,0.68) model for the BDI forecasting task. The backward horizon is taken as q = 13 periods.

## 4. The empirical results and validation

The proposed models, FILF (14,1,0.68) and E-FILF (14,1,0.68,13) provide higher accuracy than classical time series methods, and also improve the problem of unprocessed cyclical factors that is detected on error rates. The E-FILF model results also point out a white noise error pattern (Fig. 5).
The MAPE results of the proposed models and benchmark methods are compared to validate the FILF algorithm. The methods of Chen (1996) and Yu (2005) are selected as benchmark methods, and the forecasting task is performed for the same dataset. The weights of FLRs are defined as their density of existence for the method of Yu (2005). Table 6 indicates the overall performance of the methods. The FILF (14,1,0.68) model ensured the priority among the benchmark methods. The E-FILF (14,1,0.68,13) model provided a minor improvement upon the FILF model.
The analysis of errors of the methods also gives us considerable information. For example, the last 5-6 periods of the raw data have intensive decline, and although most of the methods are inferior, the FILF family models never indicate an APE higher than 1.0. Nevertheless, the benchmark methods APE score can reach over 1.5 in some periods, and particularly in recession periods (2001-2002). These methods also occasionally have a nonstop high-level APE score (Fig. 6). One of the original aspects of the FILF algorithm is the latest value adjustment. Fig. 7 shows the curve of actual data and the proposed methods.

[Table 7: The mean error rates of the methods.]
[Figure 7: The curves of the actual data and the FILF and E-FILF models for BDI.]

For white noise validation of errors, variances and means (a simple average) of the methods are compared. The results of the variance comparison of the benchmark methods' error rates indicated about 0.07 variance scores. The FILF (14,1,0.68) and the E-FILF (14,1,0.68,13) models have 0.02 variance scores. The proposed models provide better results in that the error rates follow a white noise variance standard (zero variance) more than the benchmark methods. The comparison of the mean of errors pointed out the superiority of the proposed methods as well. Table 7 shows the result of the mean error rates, which also highlights the proposed models.

## 5. Conclusion

In this paper, the fuzzy integrated logical forecasting algorithm and its error corrected extension are presented. Although the fuzzy time series forecasting methodology has many advantages over the conventional econometric approaches, some unique techniques of time series analysis can improve our understanding of time series analysis for fuzzy extended design. The traditional benchmark method of econometrics is Naïve-type forecasting, which is based on the assumption that the forecasted value of time t equals the value of time t - 1. From this origin of time series, this study extended the recent literature by applying the adjustment function of the latest value adjustment. The FILF algorithm improved prediction accuracy compared to most of the experimental studies on the BDI.
The second important improvement of the FTS is provided by an error correction function. Although the empirical work does not strongly require an error correction, as an illustrative example we performed the E-FILF procedure, and a minor improvement was indicated. However, the E-FILF algorithm can improve the accuracy of the series that have correlated, or pattern evident error rates.

The development of the E-FILF model can be based on more complicated and intelligent algorithms for modelling errors. For example, a fuzzy extended error model may be implemented for executing error patterns. In most cases, error pattern encapsulates various information which is still remaining out of the model.

## References

Ariel, A. (1989). Delphi forecast of the dry bulk shipping industry in the year 2000. Maritime Policy and Management, 16, 305-336.
Beenstock, M., & Vergottis, A. (1993). Econometric modelling of world shipping. London: Chapman and Hall.
Box, G. E. P., & Jenkins, G. M. (1976) (Time series analysis: Forecasting and control). Oakland CA: Holden-Day [Revised edition].
Bowerman, B. L., & O'Connell, R. T. (1979). Time series and forecasting: An applied approach. New York: Duxbury Press.
Charemza, W., & Gronicki, M. (1981). An econometric model of world shipping and shipbuilding. Maritime Policy and Management, 8, 21-30.
Chen, S. M. (1996). Forecasting enrolments based on fuzzy time series. Fuzzy Sets and Systems, 81, 311-319.
... (and other references as listed in the OCR)
